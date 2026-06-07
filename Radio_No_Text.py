#!/usr/bin/env python3
import os
import math
import time
import signal
import threading
import asyncio
from collections import deque
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly
from rtlsdr import RtlSdr

import discord
import discord.opus
from discord.ext import commands
from dotenv import load_dotenv


# ============================================================
# USER SETTINGS
# ============================================================

# Default station: 90.1 MHz
STATION_FREQ_HZ = 90_100_000
TUNE_OFFSET_HZ = 0
current_station_freq_hz = STATION_FREQ_HZ

# Use a rate your RTL-SDR accepted before.
# 768000 caused: "Could not set sample rate"
RF_SAMPLE_RATE = 1_024_000
IF_RATE = 256_000
AUDIO_RATE = 48_000

TUNER_GAIN = 25
FREQ_PPM = 0

DEEMPH_TAU = 75e-6
AUDIO_GAIN_DB = 6

IQ_CHUNK_SAMPLES = 262_144
MAX_BUFFER_CHUNKS = 240
TRIM_TO_CHUNKS = 120

WFM_CHAN_CUTOFF_HZ_INIT = 100_000

BOT_PREFIX = "!"


# ============================================================
# OPUS LOADING FOR DISCORD VOICE
# ============================================================

try:
    if not discord.opus.is_loaded():
        discord.opus.load_opus("libopus.so.0")
except Exception as e:
    print(f"Warning: could not load Opus automatically: {e}")
    print("Install Opus with: sudo apt install -y libopus0 libopus-dev")


# ============================================================
# DSP HELPERS
# ============================================================

def lpf(fs, cutoff_hz, taps=65):
    return firwin(taps, cutoff_hz, fs=fs)


def fir_filter(x, taps, state):
    y, zf = lfilter(taps, 1.0, x, zi=state)
    return y, zf


def rational_resample_ratio(fs_in, fs_out, max_den=512):
    frac = Fraction(fs_out, fs_in).limit_denominator(max_den)
    return frac.numerator, frac.denominator


class Deemphasis:
    def __init__(self, fs, tau):
        self.a = float(np.exp(-1.0 / (fs * tau)))
        self.b = [1.0 - self.a]
        self.a_den = [1.0, -self.a]
        self.zi = np.array([0.0], dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, zf = lfilter(
            self.b,
            self.a_den,
            x.astype(np.float32, copy=False),
            zi=self.zi,
        )
        self.zi = zf
        return y.astype(np.float32)


class AudioAGC:
    def __init__(self, fs=AUDIO_RATE, target=0.16, attack_ms=25, release_ms=250):
        self.target = float(target)
        self.attack_a = math.exp(-1.0 / (fs * (attack_ms / 1000.0)))
        self.release_a = math.exp(-1.0 / (fs * (release_ms / 1000.0)))
        self.env = 1e-3

    def process(self, stereo: np.ndarray) -> np.ndarray:
        if stereo.size == 0:
            return stereo

        env_in = np.max(np.abs(stereo), axis=1)

        for v in env_in:
            a = self.attack_a if v > self.env else self.release_a
            self.env = a * self.env + (1.0 - a) * float(v)

        gain = self.target / (self.env + 1e-9)
        gain = float(np.clip(gain, 0.1, 12.0))

        return (stereo * gain).astype(np.float32)


# ============================================================
# LIGHTER MONO WFM DEMODULATOR
# ============================================================

class WFMMonoDemod:
    """
    Lighter mono FM demodulator for Raspberry Pi.

    This avoids stereo pilot/subcarrier decoding, which is much heavier
    and can make Discord audio choppy on lower-power devices.
    """

    def __init__(self, chan_cutoff_hz: int):
        cutoff_hz = int(np.clip(chan_cutoff_hz, 40_000, 160_000))

        self.chan_taps = lpf(RF_SAMPLE_RATE, cutoff_hz, taps=65)
        self.chan_state = np.zeros(len(self.chan_taps) - 1, dtype=np.complex64)

        self.audio_taps = lpf(IF_RATE, 15_000, taps=65)
        self.audio_state = np.zeros(len(self.audio_taps) - 1, dtype=np.float32)

        self.prev_iq = None

        self.up_if, self.down_if = rational_resample_ratio(RF_SAMPLE_RATE, IF_RATE)
        self.up_a, self.down_a = rational_resample_ratio(IF_RATE, AUDIO_RATE)

        self.deemph = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.agc = AudioAGC(fs=AUDIO_RATE)
        self.audio_gain = float(10 ** (AUDIO_GAIN_DB / 20.0))

    def fm_discriminator(self, iq_if: np.ndarray) -> np.ndarray:
        if len(iq_if) == 0:
            return np.zeros((0,), dtype=np.float32)

        if self.prev_iq is None:
            self.prev_iq = iq_if[0]

        x_prev = np.concatenate(([self.prev_iq], iq_if[:-1]))
        self.prev_iq = iq_if[-1]

        d = iq_if * np.conj(x_prev)
        return np.angle(d).astype(np.float32)

    def process_block(self, iq: np.ndarray) -> np.ndarray:
        iq = iq.astype(np.complex64, copy=False)

        iq_f, self.chan_state = fir_filter(iq, self.chan_taps, self.chan_state)
        iq_if = resample_poly(iq_f, self.up_if, self.down_if).astype(np.complex64)

        fm = self.fm_discriminator(iq_if)

        audio, self.audio_state = fir_filter(fm, self.audio_taps, self.audio_state)
        audio = resample_poly(audio, self.up_a, self.down_a).astype(np.float32)

        audio = self.deemph.process(audio)

        # Discord expects stereo PCM, so duplicate mono into left/right.
        stereo = np.column_stack([audio, audio]).astype(np.float32)

        stereo = self.agc.process(stereo)
        stereo *= self.audio_gain

        peak = float(np.max(np.abs(stereo)) + 1e-9) if stereo.size else 1e-9
        if peak > 0.98:
            stereo *= 0.98 / peak

        return stereo.astype(np.float32)


# ============================================================
# DISCORD AUDIO SOURCE
# ============================================================

class DiscordRadioSource(discord.AudioSource):
    """
    Discord expects 20ms chunks of 48kHz stereo signed 16-bit PCM.

    48,000 Hz * 0.020 sec = 960 frames
    960 frames * 2 channels * 2 bytes = 3840 bytes
    """

    def __init__(self, audio_buffer, buf_lock, stop_event):
        self.audio_buffer = audio_buffer
        self.buf_lock = buf_lock
        self.stop_event = stop_event

        self.leftover = np.zeros((0, 2), dtype=np.float32)
        self.frames_per_read = 960
        self.last_low_buffer_print = 0.0

    def read(self):
        if self.stop_event.is_set():
            return b""

        out = np.zeros((self.frames_per_read, 2), dtype=np.float32)
        need = self.frames_per_read
        idx = 0

        if len(self.leftover) > 0:
            take = min(need, len(self.leftover))
            out[idx:idx + take] = self.leftover[:take]
            self.leftover = self.leftover[take:]
            idx += take
            need -= take

        while need > 0:
            with self.buf_lock:
                chunk = self.audio_buffer.popleft() if self.audio_buffer else None
                buffer_len = len(self.audio_buffer)

            if chunk is None:
                now = time.time()
                if now - self.last_low_buffer_print > 2.0:
                    print("Audio buffer empty or low")
                    self.last_low_buffer_print = now
                break

            take = min(need, len(chunk))
            out[idx:idx + take] = chunk[:take]
            idx += take
            need -= take

            if take < len(chunk):
                self.leftover = chunk[take:]
                break

        out = np.clip(out, -1.0, 1.0)
        pcm16 = (out * 32767.0).astype(np.int16)

        return pcm16.tobytes()

    def is_opus(self):
        return False


# ============================================================
# RADIO RUNNER
# ============================================================

class RadioRunner:
    def __init__(self, text_channel, bot_loop):
        self.text_channel = text_channel
        self.bot_loop = bot_loop

        self.stop_event = threading.Event()

        self.buf_lock = threading.Lock()
        self.audio_buffer = deque()

        self.sdr = None
        self.radio_thread = None

        self.last_peak_print = 0.0

    def get_audio_source(self):
        return DiscordRadioSource(
            self.audio_buffer,
            self.buf_lock,
            self.stop_event,
        )

    def start(self):
        self.stop_event.clear()

        self.radio_thread = threading.Thread(
            target=self.radio_worker,
            daemon=True,
        )
        self.radio_thread.start()

    def stop(self):
        self.stop_event.set()

        if self.sdr is not None:
            try:
                self.sdr.cancel_read_async()
            except Exception:
                pass

            time.sleep(0.5)

            try:
                self.sdr.close()
            except Exception:
                pass

            time.sleep(1.0)

            self.sdr = None

        with self.buf_lock:
            self.audio_buffer.clear()

    def send_text_to_discord(self, text):
        async def send_msg():
            await self.text_channel.send(f"📻 **Radio:** {text}")

        try:
            asyncio.run_coroutine_threadsafe(send_msg(), self.bot_loop)
        except Exception as e:
            print(f"Could not send Discord message: {e}")

    def buffer_count(self):
        with self.buf_lock:
            return len(self.audio_buffer)

    def radio_worker(self):
        global current_station_freq_hz

        try:
            self.send_text_to_discord(
                f"Tuning to `{current_station_freq_hz / 1_000_000:.3f} MHz`..."
            )

            demod = WFMMonoDemod(WFM_CHAN_CUTOFF_HZ_INIT)

            self.sdr = RtlSdr()
            self.sdr.sample_rate = RF_SAMPLE_RATE
            self.sdr.center_freq = current_station_freq_hz + TUNE_OFFSET_HZ
            self.sdr.gain = TUNER_GAIN

            if FREQ_PPM:
                try:
                    self.sdr.freq_correction = int(FREQ_PPM)
                except Exception as e:
                    self.send_text_to_discord(
                        f"Warning: could not set PPM correction: `{e}`"
                    )

            def rtl_callback(iq, _ctx):
                if self.stop_event.is_set():
                    return

                try:
                    stereo = demod.process_block(iq)

                    now = time.time()
                    if now - self.last_peak_print > 5.0:
                        peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
                        with self.buf_lock:
                            blen = len(self.audio_buffer)
                        print(f"audio peak: {peak:.4f}, buffer chunks: {blen}")
                        self.last_peak_print = now

                    with self.buf_lock:
                        self.audio_buffer.append(stereo)

                        if len(self.audio_buffer) > MAX_BUFFER_CHUNKS:
                            while len(self.audio_buffer) > TRIM_TO_CHUNKS:
                                self.audio_buffer.popleft()

                except Exception as e:
                    print(f"radio callback error: {e}")

            self.sdr.read_samples_async(rtl_callback, IQ_CHUNK_SAMPLES)

        except Exception as e:
            self.send_text_to_discord(f"Radio error: `{e}`")


# ============================================================
# DISCORD BOT
# ============================================================

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in .env file")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

radio_runner = None


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.command()
async def join(ctx):
    if ctx.author.voice is None:
        await ctx.send("Join a voice channel first.")
        return

    channel = ctx.author.voice.channel

    if ctx.voice_client is not None:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()

    await ctx.send(f"Joined **{channel.name}**.")


@bot.command()
async def radio(ctx):
    global radio_runner

    if ctx.author.voice is None:
        await ctx.send("Join a voice channel first.")
        return

    if ctx.voice_client is None:
        await ctx.author.voice.channel.connect()

    if radio_runner is not None:
        await ctx.send("Radio is already running.")
        return

    radio_runner = RadioRunner(ctx.channel, bot.loop)

    # Start SDR first so the buffer has time to fill.
    radio_runner.start()

    await ctx.send("Buffering radio audio...")
    await asyncio.sleep(2.5)

    source = radio_runner.get_audio_source()

    ctx.voice_client.play(
        source,
        after=lambda e: print(f"Discord voice playback error: {e}") if e else None,
    )

    await ctx.send(
        f"Started radio at **{current_station_freq_hz / 1_000_000:.3f} MHz**."
    )


@bot.command()
async def stopradio(ctx):
    global radio_runner

    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None

    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()

    await ctx.send("Stopped radio.")


@bot.command()
async def leave(ctx):
    global radio_runner

    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None

    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")


@bot.command()
async def tune(ctx, freq_mhz: float):
    """
    Safer retune:
    stop Discord playback, close RTL-SDR, wait, reopen at new frequency.
    This avoids live retuning errors like i2c failed=-6 / LIBUSB busy.
    """
    global radio_runner, current_station_freq_hz

    if freq_mhz < 50 or freq_mhz > 110:
        await ctx.send("Please enter a normal FM station, like `!tune 101.1`.")
        return

    current_station_freq_hz = int(freq_mhz * 1_000_000)

    was_running = radio_runner is not None

    if not was_running:
        await ctx.send(
            f"Station set to **{freq_mhz:.3f} MHz**. Start it with `!radio`."
        )
        return

    if ctx.voice_client is None:
        await ctx.send("I am not connected to voice.")
        return

    await ctx.send(f"Retuning to **{freq_mhz:.3f} MHz**...")

    try:
        if ctx.voice_client.is_playing():
            ctx.voice_client.stop()

        radio_runner.stop()
        radio_runner = None

        await asyncio.sleep(2.0)

        radio_runner = RadioRunner(ctx.channel, bot.loop)
        radio_runner.start()

        await ctx.send("Buffering radio audio...")
        await asyncio.sleep(2.5)

        source = radio_runner.get_audio_source()

        ctx.voice_client.play(
            source,
            after=lambda e: print(f"Discord voice playback error: {e}") if e else None,
        )

        await ctx.send(f"Retuned to **{freq_mhz:.3f} MHz**.")

    except Exception as e:
        await ctx.send(f"Could not tune radio: `{e}`")


@bot.command()
async def status(ctx):
    global radio_runner

    if radio_runner is None:
        await ctx.send("Radio is not running.")
        return

    vc = ctx.voice_client

    if vc is None:
        await ctx.send("Radio runner exists, but I am not connected to voice.")
        return

    await ctx.send(
        f"Radio running: `{radio_runner is not None}`\n"
        f"Voice connected: `{vc.is_connected()}`\n"
        f"Voice playing: `{vc.is_playing()}`\n"
        f"Audio buffer chunks: `{radio_runner.buffer_count()}`"
    )


@bot.command()
async def shutdown(ctx):
    global radio_runner

    await ctx.send("Shutting down radio bot...")

    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None

    if ctx.voice_client:
        try:
            if ctx.voice_client.is_playing():
                ctx.voice_client.stop()
            await ctx.voice_client.disconnect()
        except Exception:
            pass

    await bot.close()


def shutdown_handler(*_args):
    global radio_runner

    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


bot.run(TOKEN)
