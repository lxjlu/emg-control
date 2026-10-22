# -*- coding: utf-8 -*-
"""Go2 轮式（Go2-W）适配器。

与原 udp_go2_bridge_v2.py 的 go2w 分支一致：动之前必须先 BalanceStand
（等效于 Go2w_sport_client 里按 9），否则 Move() 不会让轮足动起来。
"""

import time

from ..protocol import ACTION_MOVE
from .go2 import Go2Adapter


class Go2WAdapter(Go2Adapter):
    name = "go2w"
    display = "Go2 轮式（Go2-W）"

    max_vx = 0.5
    max_wz = 1.0

    def __init__(self, interface="eth0", robot_ip=None, balance_settle=1.5,
                 dry_run=False, log=print):
        super().__init__(interface=interface, robot_ip=robot_ip, dry_run=dry_run, log=log)
        self.balance_settle = float(balance_settle)
        self._prepared = False

    # ------------------------------------------------------------------
    def _ensure_prepared(self):
        """第一次运动前进入轮式平衡站立。"""
        if self._prepared:
            return
        self._prepared = True
        if self.dry_run:
            self.log("[dry-run] Go2-W BalanceStand（等效按 9）")
            return
        self.log("Go2-W：BalanceStand，进入轮式平衡站立…")
        self._client.BalanceStand()
        time.sleep(self.balance_settle)
        self.log("Go2-W：平衡站立完成")

    def apply(self, action, args):
        if action != ACTION_MOVE:
            return
        self._ensure_prepared()
        super().apply(action, args)
