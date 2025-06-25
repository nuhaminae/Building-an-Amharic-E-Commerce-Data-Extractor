import os
import csv
from dotenv import load_dotenv
from telethon import TelegramClient
import asyncio

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.abspath(os.path.join('..')), '.env'))

# Get Telegram API credentials from environment variables
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')

# Check if API credentials are provided
if not api_id or not api_hash:
    raise ValueError("Missing Telegram API credentials. Check your .env file.")

# Define path for the Telegram session file
session_path = os.path.join('..', 'data/raw', 'scraping_session')
# Initalise Telegram client
client = TelegramClient(session_path, api_id, api_hash)

async def scrape_channel(client, channel_username, writer):
    """
    Scrapes messages from a given Telegram channel and writes them to a CSV.

    Args:
        client (TelegramClient): The Telegram client.
        channel_username (str): The username of the Telegram channel.
        writer (csv.writer): The CSV writer object.

    """

    # Get the channel entity
    entity = await client.get_entity(channel_username)
    channel_title = entity.title # Get the channel title
    
    # Iterate through messages in the channel (up to a limit)
    async for message in client.iter_messages(entity, limit=10000):
        # Write message data to the CSV file
        writer.writerow([
            channel_title, channel_username,
            message.id, message.message,
            message.date,
            message.views or 0  # Fallback if views is None
        ])

async def main():
    """
    Main function to initialize the client, set up directories,
    and scrape data from a list of channels.
    """
    
    print("\nStarting Telegram client...")
    await client.start() # Start the telegram client
    print("\nTelegram client started.")

    # Define root directory for data storage
    data_root = os.path.abspath(os.path.join('..', 'data/raw'))
    os.makedirs(data_root, exist_ok=True) # Create data directory if it doesn't exist

    # List of Telegram channels to scrape
    channels = [
        '@gebeyaadama', '@kuruwear', '@Leyueqa',
        '@MerttEka', '@qnashcom', '@Shewabrand'
    ]

    # Iterate through each channel
    for channel in channels:
        # Define paths for CSV
        csv_path = os.path.join(data_root, f"{channel[1:]}.csv")

        # Skip scraping if the CSV file for this channel already exists
        if os.path.exists(csv_path):
            print(f"\n✓ {channel} already scraped — skipping.")
            continue

        print(f"\nScraping data from {channel}")
        try:
            # Open the CSV file for writing 
            with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file) # Create a CSV writer
                # Write the header row to the CSV
                writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Views'])

                # Scrape data from the channel
                await scrape_channel(client, channel, writer)
            print(f"\n✓ Finished scraping {channel}")
        except Exception as e:
            # Catch and print any errors during scraping
            print(f"\n⚠ Skipping {channel} due to error: {e}")

# Run the main asynchronous function
if __name__ == "__main__":
    asyncio.run(main())