#!/usr/bin/env bash
set -euo pipefail
umask 077

# 仅安装当前账号的 CLI 和用户服务。配置存在时继续沿用。
spool_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
spool_build="$(mktemp -d)"
trap 'rm -f -- "$spool_build/spool" && rmdir -- "$spool_build"' EXIT
python3 -m zipapp "$spool_source/app" -o "$spool_build/spool" -p '/usr/bin/env python3'
mkdir -p "$HOME/.local/bin" "$HOME/.config/spool" "$HOME/.config/systemd/user"
spool_target="$(mktemp "$HOME/.local/bin/.spool.XXXXXXXX")"
trap 'rm -f -- "$spool_target" "$spool_build/spool" && rmdir -- "$spool_build"' EXIT
install -m 0755 "$spool_build/spool" "$spool_target"
mv -f -- "$spool_target" "$HOME/.local/bin/spool"
if [[ ! -e "$HOME/.config/spool/config.json" && ! -L "$HOME/.config/spool/config.json" ]]; then
    install -m 0600 "$spool_source/deploy/config.example.json" "$HOME/.config/spool/config.json"
fi
install -m 0644 "$spool_source/deploy/spool.service" "$HOME/.config/systemd/user/spool.service"
