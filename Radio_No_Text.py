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
from discord.ext import commands
from dotenv import load_dotenv


# ============================================================
# USER SETTINGS
# ============================================================

STATION_FREQ_HZ = 90_098_000
TUNE_OFFSET_HZ = 0
CENTER_FREQ_HZ = STATION_FREQ_HZ + TUNE_OFFSET_HZ

RF_SAMPLE_RATE = 1_024_000
IF_RATE = 240_000
AUDIO_RATE = 48_000

TUNER_GAIN = 25
FREQ_PPM = 0

DEEMPH_TAU = 75e-6
AUDIO_GAIN_DB = -12.3

IQ_CHUNK_SAMPLES = 262_144
MAX_BUFFER_CHUNKS = 120
TRIM_TO_CHUNKS = 40

WFM_CHAN_CUTOFF_HZ_INIT = 100_000

BOT_PREFIX = "!"


# ============================================================
# DSP HELPERS
# ============================================================

def lpf(fs, cutoff_hz, taps=129):
    return firwin(taps, cutoff_hz, fs=fs)


def bpf(fs, lo_hz, hi_hz, taps=257):
    return firwin(taps, [lo_hz, hi_hz], pass_zero=False, fs=fs)


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
    def __init__(self, fs=AUDIO_RATE, target=0.12, attack_ms=25, release_ms=250):
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
        gain = float(np.clip(gain, 0.1, 10.0))

        return (stereo * gain).astype(np.float32)


# ============================================================
# WFM STEREO DEMODULATOR
# ============================================================

class WFMStereoOffset:
    def __init__(self, chan_cutoff_hz: int):
        self.cfg_lock = threading.Lock()
        self._build_chan_filter(chan_cutoff_hz)

        self.mono_taps = lpf(IF_RATE, 15_000, taps=129)
        self.mono_state = np.zeros(len(self.mono_taps) - 1, dtype=np.float32)

        self.pilot_taps = bpf(IF_RATE, 18_500, 19_500, taps=257)
        self.pilot_state = np.zeros(len(self.pilot_taps) - 1, dtype=np.float32)

        self.sub_taps = bpf(IF_RATE, 36_000, 40_000, taps=257)
        self.sub_state = np.zeros(len(self.sub_taps) - 1, dtype=np.float32)

        self.stereo_taps = bpf(IF_RATE, 23_000, 53_000, taps=257)
        self.stereo_state = np.zeros(len(self.stereo_taps) - 1, dtype=np.float32)

        self.lr_lpf_taps = lpf(IF_RATE, 15_000, taps=129)
        self.lr_lpf_state = np.zeros(len(self.lr_lpf_taps) - 1, dtype=np.float32)

        self.prev_iq = None

        self.phase = 0.0
        self.step = 2.0 * np.pi * (TUNE_OFFSET_HZ / RF_SAMPLE_RATE)

        self.up_if, self.down_if = rational_resample_ratio(RF_SAMPLE_RATE, IF_RATE)
        self.up_a, self.down_a = rational_resample_ratio(IF_RATE, AUDIO_RATE)

        self.deemph_L = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.deemph_R = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.agc = AudioAGC(fs=AUDIO_RATE)

        self.audio_gain = float(10 ** (AUDIO_GAIN_DB / 20.0))

    def _build_chan_filter(self, cutoff_hz: int):
        cutoff_hz = int(np.clip(cutoff_hz, 40_000, 160_000))
        self.chan_cutoff_hz = cutoff_hz
        self.chan_taps = lpf(RF_SAMPLE_RATE, cutoff_hz, taps=129)
        self.chan_state = np.zeros(len(self.chan_taps) - 1, dtype=np.complex64)

    def mix_offset_to_baseband(self, iq: np.ndarray) -> np.ndarray:
        if TUNE_OFFSET_HZ == 0:
            return iq.astype(np.complex64, copy=False)

        n = np.arange(len(iq), dtype=np.float32)
        ph = self.phase + self.step * n
        osc = np.exp(1j * ph).astype(np.complex64)
        self.phase = float((self.phase + self.step * len(iq)) % (2.0 * np.pi))

        return iq.astype(np.complex64, copy=False) * osc

    def fm_discriminator(self, iq_if: np.ndarray) -> np.ndarray:
        if self.prev_iq is None:
            self.prev_iq = iq_if[0]

        x_prev = np.concatenate(([self.prev_iq], iq_if[:-1]))
        self.prev_iq = iq_if[-1]

        d = iq_if * np.conj(x_prev)
        return np.angle(d).astype(np.float32)

    def process_block(self, iq: np.ndarray) -> np.ndarray:
        iq = self.mix_offset_to_baseband(iq)

        with self.cfg_lock:
            iq_f, self.chan_state = fir_filter(iq, self.chan_taps, self.chan_state)

        iq_if = resample_poly(iq_f, self.up_if, self.down_if).astype(np.complex64)
        fm = self.fm_discriminator(iq_if)

        mono, self.mono_state = fir_filter(fm, self.mono_taps, self.mono_state)
        pilot, self.pilot_state = fir_filter(fm, self.pilot_taps, self.pilot_state)

        sub38_raw = pilot * pilot
        sub38, self.sub_state = fir_filter(sub38_raw, self.sub_taps, self.sub_state)

        rms = float(np.sqrt(np.mean(sub38 * sub38)) + 1e-9)
        sub38 = sub38 / rms

        stereo_band, self.stereo_state = fir_filter(fm, self.stereo_taps, self.stereo_state)
        lr = stereo_band * (2.0 * sub38)
        lr, self.lr_lpf_state = fir_filter(lr, self.lr_lpf_taps, self.lr_lpf_state)

        left = 0.5 * (mono + lr)
        right = 0.5 * (mono - lr)

        left_a = resample_poly(left, self.up_a, self.down_a).astype(np.float32)
        right_a = resample_poly(right, self.up_a, self.down_a).astype(np.float32)

        left_a = self.deemph_L.process(left_a)
        right_a = self.deemph_R.process(right_a)

        stereo = np.column_stack([left_a, right_a]).astype(np.float32)
        stereo = self.agc.process(stereo)
        stereo *= self.audio_gain

        peak = float(np.max(np.abs(stereo)) + 1e-9)

        if peak > 0.98:
            stereo *= 0.98 / peak

        return stereo


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

            if chunk is None:
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
    
            try:
                self.sdr.close()
            except Exception:
                pass
    
            self.sdr = None
    
        with self.buf_lock:
            self.audio_buffer.clear()
    
    def send_text_to_discord(self, text):
        async def send_msg():
            await self.text_channel.send(f"📻 **Radio:** {text}")

        asyncio.run_coroutine_threadsafe(send_msg(), self.bot_loop)


    def tune(self, freq_mhz: float):
        """
        Change the radio station while the SDR is running.
        Example: tune(101.1) -> 101.1 MHz
        """
        new_freq_hz = int(freq_mhz * 1_000_000)

        if self.sdr is None:
            raise RuntimeError("SDR is not running yet.")

        self.sdr.center_freq = new_freq_hz

        self.send_text_to_discord(
            f"Tuned to **{freq_mhz:.3f} MHz**"
        )


    def radio_worker(self):
        try:
            self.send_text_to_discord(
                f"Tuning to `{STATION_FREQ_HZ / 1_000_000:.3f} MHz`..."
            )

            demod = WFMStereoOffset(WFM_CHAN_CUTOFF_HZ_INIT)

            self.sdr = RtlSdr()
            self.sdr.sample_rate = RF_SAMPLE_RATE
            self.sdr.center_freq = CENTER_FREQ_HZ
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

                    with self.buf_lock:
                        self.audio_buffer.append(stereo)

                        if len(self.audio_buffer) > MAX_BUFFER_CHUNKS:
                            while len(self.audio_buffer) > TRIM_TO_CHUNKS:
                                self.audio_buffer.popleft()

                except Exception:
                    pass

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

    source = radio_runner.get_audio_source()

    ctx.voice_client.play(
        source,
        after=lambda e: print(f"Discord voice playback error: {e}") if e else None,
    )

    radio_runner.start()

    await ctx.send(
        f"Started radio at **{STATION_FREQ_HZ / 1_000_000:.3f} MHz**."
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
    global radio_runner

    if radio_runner is None:
        await ctx.send("Radio is not running. Start it first with `!radio`.")
        return

    if freq_mhz < 50 or freq_mhz > 110:
        await ctx.send("Please enter a normal FM station, like `!tune 101.1`.")
        return

    try:
        radio_runner.tune(freq_mhz)
        await ctx.send(f" Tuning to **{freq_mhz:.3f} MHz**...")
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
        f"Voice playing: `{vc.is_playing()}`"
    )

@bot.command()
async def shutdown(ctx):
    global radio_runner

    await ctx.send("Shutting down radio bot...")

    # Stop radio SDR/thread
    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None

    # Stop Discord voice playback
    if ctx.voice_client:
        try:
            if ctx.voice_client.is_playing():
                ctx.voice_client.stop()
            await ctx.voice_client.disconnect()
        except Exception:
            pass

    # Close Discord bot cleanly
    await bot.close()

def shutdown_handler(*_args):
    global radio_runner

    if radio_runner is not None:
        radio_runner.stop()
        radio_runner = None

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


bot.run(TOKEN)