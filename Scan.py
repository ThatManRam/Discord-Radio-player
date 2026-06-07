import discord
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("DISCORD_TOKEN")


print("Token loaded:", token is not None)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

ALLOWED_USER_IDS = [
    user_id.strip()
    for user_id in os.getenv("ALLOWED", "").split(",")
    if user_id.strip()
]




@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):

    if message.author == client.user:
        return
    
    if str(message.author.id) not in ALLOWED_USER_IDS:
        print("User not allowed")
        return

    if message.content.lower().startswith("!stop"):
        await message.channel.send("Scanning ended")
        exit()
        return


    if "kitten" not in message.content.lower():
        return

    async with message.channel.typing():
        try:
            history_messages = await get_discord_history(message, limit=2)

            reply = await asyncio.to_thread(
                ask_ollama_with_history,
                history_messages
            )

            if len(reply) > 1900:
                reply = reply[:1900] + "..."

            await message.channel.send(reply)

        except Exception as e:
            await message.channel.send(f"Error talking to Ollama: `{e}`")


client.run(token)