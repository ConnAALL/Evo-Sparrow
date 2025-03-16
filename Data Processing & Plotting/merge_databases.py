"""
Simple script to merge multiple databases into a single database without duplicates.
"""
import sqlite3

def merge_databases(db_files, merged_db="merged_database.db", table_name="results"):
    """Merge multiple SQLite .db files into a new single database without duplicates."""
    
    if not db_files:
        print("No database files provided.")
        return
    
    # Create a new merged database
    merged_conn = sqlite3.connect(merged_db)
    merged_cursor = merged_conn.cursor()

    first_db = db_files[0]

    # Copy schema from the first database
    with sqlite3.connect(first_db) as conn:
        schema = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()
        if schema:
            merged_cursor.execute(schema[0])
            print(f"Created table {table_name} in {merged_db}.")
        else:
            print(f"Table {table_name} not found in {first_db}. Skipping merge.")
            return

    existing_rows = set()

    # Iterate over each database and merge into the new one
    for db_file in db_files:
        print(f"Merging {db_file} into {merged_db}...")
        src_conn = sqlite3.connect(db_file)
        src_cursor = src_conn.cursor()

        # Fetch all data from the table
        try:
            src_cursor.execute(f"SELECT * FROM {table_name}")
            rows = src_cursor.fetchall()
            if not rows:
                print(f"No data found in {table_name} of {db_file}. Skipping.")
                continue
            
            # Get column names dynamically
            src_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in src_cursor.fetchall()]
            column_count = len(columns)
            placeholders = ", ".join(["?"] * column_count)  

            # Track duplicates
            duplicate_count = 0
            unique_rows = []
            
            for row in rows:
                if row in existing_rows:
                    duplicate_count += 1
                else:
                    existing_rows.add(row)
                    unique_rows.append(row)

            # Insert unique rows into the new database
            if unique_rows:
                insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                merged_cursor.executemany(insert_query, unique_rows)
                merged_conn.commit()

            print(f"Successfully merged {len(unique_rows)} unique rows from {db_file}.")
            if duplicate_count > 0:
                print(f"Skipped {duplicate_count} duplicate rows from {db_file}.")

        except sqlite3.Error as e:
            print(f"Error merging {db_file}: {e}")

        finally:
            src_conn.close()

    merged_conn.close()
    print(f"All databases merged successfully into {merged_db}.")

# List of the .db files to merge
db_files = ["250308_01Trials_200Batch.db",
            "250308_02Trials_200Batch.db",
            "250308_18Trials_200Batch.db",
            "250308_07Trials_200Batch.db",
            "250309_15Trials_200Batch.db",
            "250309_01Trials_200Batch.db",
            "250309_10Trials_200Batches.db",
            "250309_07Trials_200Batch.db",
            "250310_14Trials_200Batches.db",
            "250310_07Trials_200Batches.db",
            "250311_10Trials_200Batches.db",
            "250311_11Trials_200Batches.db",
            "250311_08Runs_200Batches.db"]

merge_databases(db_files, "merged_output.db")
