#!/bin/bash
sqlite3 llm_cache.db "SELECT strftime('%Y-%m-%d %H:00', created_at) AS hour, count(*) AS calls FROM llm_cache WHERE datetime(created_at) >= datetime('now', '-${NUM_HOURS} hours') GROUP BY hour ORDER BY hour;"

