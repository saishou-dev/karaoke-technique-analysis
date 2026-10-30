#!/bin/bash
# launch_wd.sh ＝ 番犬だけを分離起動（root で呼ぶ）
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
cat > /tmp/wd_inner.sh <<EOS
exec su toshiaki -c "bash '$D/watchdog_scan.sh' 10"
EOS
setsid nohup bash /tmp/wd_inner.sh < /dev/null > /dev/null 2>&1 &
sleep 5; echo launched
