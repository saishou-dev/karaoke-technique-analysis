#!/bin/bash
D=/mnt/g/研究アーカイブ/成果物/動画解析/unread_fix_validate_20261030
cat > /tmp/v1_inner.sh <<EOS
echo \$\$ > /sys/fs/cgroup/research/cgroup.procs 2>/dev/null || echo "cgroup: 入れず" >> "$D/v1_pipeline.log"
exec su toshiaki -c "bash '$D/v1_pipeline.sh' >> '$D/v1_pipeline.log' 2>&1"
EOS
setsid nohup bash /tmp/v1_inner.sh < /dev/null > /dev/null 2>&1 &
sleep 5; echo launched
