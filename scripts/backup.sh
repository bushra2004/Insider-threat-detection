#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$TIMESTAMP"

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker-compose exec db pg_dump -U postgres threatdb > $BACKUP_DIR/threatdb_backup.sql

# Backup Redis
docker-compose exec redis redis-cli SAVE
docker-compose cp redis:/data/dump.rdb $BACKUP_DIR/redis_backup.rdb

echo "✅ Backup completed: $BACKUP_DIR"