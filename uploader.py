import sqlite3
import time
from supabase import create_client, Client
from httpx import HTTPError

# --- CONFIG ---
DB_PATH = "tealscript.db"
SUPABASE_URL = "https://toqvsuthxooqjelcayhm.supabase.co"
SUPABASE_KEY =  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRvcXZzdXRoeG9vcWplbGNheWhtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDcwNzE4MCwiZXhwIjoyMDcwMjgzMTgwfQ.C9FGRTS83LdyhV0hhvgMzlHfHzflsUcvceSGIzkRcNU"  # Use service role key for inserts
BATCH_SIZE = 50  # smaller batches to avoid SSL drop
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds

# --- Connect to Supabase ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_table(table_name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    total = len(rows)
    print(f"Uploading {total} rows to {table_name}...")

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        records = [dict(zip(columns, row)) for row in batch]

        for attempt in range(MAX_RETRIES):
            try:
                supabase.table(table_name).insert(records).execute()
                break
            except HTTPError as e:
                print(f"⚠️ Batch {i//BATCH_SIZE + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print("❌ Max retries reached, skipping batch.")
        print(f"✅ {min(i + BATCH_SIZE, total)} / {total} rows uploaded.")

    conn.close()

if __name__ == "__main__":
    upload_table("documents")
    upload_table("chunks")
    print("🎯 Upload complete")
