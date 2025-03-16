"""
File for managing the database of game results and logging of run parameters.
"""

import sqlite3, os, time


def setup_database(db_name="../DATA/game_results.db"):
    """
    Setup the database for storing game results.
    As the database is used by multiple processes, the WAL (Write-Ahead Logging) mode is used to allow for concurrent reads and writes.
    """
    conn = sqlite3.connect(db_name, timeout=30)  # The timeout parameters is set to 30 seconds. This is the maximum time the database will wait for a lock to be released.
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        game_id TEXT,
        player_1_reward REAL,
        player_2_reward REAL,
        player_3_reward REAL
    )
    ''')
    conn.commit()
    return conn, cursor


def insert_game_result(cursor, rewards, game_id):
    """Insert the results of a game into the database."""
    cursor.execute('''
    INSERT INTO results (game_id, player_1_reward, player_2_reward, player_3_reward)
    VALUES (?, ?, ?, ?)
    ''', (game_id, rewards[0], rewards[1], rewards[2]))


def save_game_results(state, cursor, game_id, idx):
    """Check and save results of ended games with a retry on database locks."""
    if state.terminated[idx] or state.truncated[idx]:
        retries = 5
        while retries > 0:
            try:
                insert_game_result(cursor, state.rewards[idx].tolist(), game_id)
                break  # Success, exit the loop.
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(0.1)  # Wait 100ms before retrying.
                    retries -= 1
                else:
                    raise


def create_log_file(log_file_name="../DATA/PARAMETER_LOGS.txt"):
    """Create a log file for storing the parameters of the run."""
    log_file_path = os.path.join(os.getcwd(), log_file_name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if not os.path.exists(log_file_path):  # If the file does not exist, create it.
        with open(log_file_path, 'w') as file: file.write("PARAMETER LOG\n\n")
    return log_file_path


def log_run_parameters(log_file_path, log_txt):
    """Log the parameters of the run to a log file."""

    # Create the log file if it does not exist.
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, 'a') as log_file:
        log_file.write(log_txt + '\n')
