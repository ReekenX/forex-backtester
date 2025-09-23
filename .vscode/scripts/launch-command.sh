#!/bin/bash

tmux rename-window -t${TMUX_PANE} "🟢 $1"
$2
tmux rename-window -t${TMUX_PANE} "🔴 $1"

