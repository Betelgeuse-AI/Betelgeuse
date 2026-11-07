#!/usr/bin/env bash
# Sequential, resumable, sha256-verified HF downloader (aria2c)
set -uo pipefail

REPO="unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
BASE="https://huggingface.co/$REPO"
DEST="${DEST:-$HOME/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017}"
META="$DEST/.meta"
mkdir -p "$DEST" "$META"

P="UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL"
FILES=(
  "$P-00001-of-00007.gguf" "$P-00002-of-00007.gguf" "$P-00003-of-00007.gguf"
  "$P-00004-of-00007.gguf" "$P-00005-of-00007.gguf" "$P-00006-of-00007.gguf"
  "engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file"
  "dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf"
)

R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; C=$'\e[36m'; BOLD=$'\e[1m'; DIM=$'\e[2m'; N=$'\e[0m'
if command -v sha256sum >/dev/null; then SHA256=(sha256sum); else SHA256=(shasum -a 256); fi

fsize(){ stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
ondisk(){ local k; k=$(du -k "$1" 2>/dev/null | cut -f1); echo $(( ${k:-0} * 1024 )); }
hum(){ awk -v b="$1" 'BEGIN{split("B KiB MiB GiB TiB",u," ");i=1;while(b>=1024&&i<5){b/=1024;i++}printf "%.1f %s",b,u[i]}'; }
bar(){ local p=$1 w=24 i out="" f; f=$(( p*w/100 )); for((i=0;i<w;i++)); do if((i<f)); then out+="█"; else out+="░"; fi; done; printf '%s' "$out"; }
lname(){ local n; n=$(basename "$1"); echo "${n%_file}"; }

# sets RPATH SHA SIZE (from HF LFS pointer; cached)
fetch_meta(){
  local name=$1 path=$2 f="$META/$1" try out
  if [[ -s $f ]]; then read -r RPATH SHA SIZE < "$f"; return 0; fi
  for try in "$path" "${path%_file}"; do
    out=$(curl -fsSL --retry 5 --retry-delay 3 "$BASE/raw/main/$try") || continue
    SHA=$(awk '/^oid sha256:/{print substr($2,8)}' <<<"$out")
    SIZE=$(awk '/^size /{print $2}' <<<"$out")
    if [[ -n $SHA && -n $SIZE ]]; then RPATH=$try; echo "$RPATH $SHA $SIZE" > "$f"; return 0; fi
  done
  return 1
}

state(){
  local name=$1 f="$DEST/$1"
  [[ -f $META/$name.ok ]] && { echo VERIFIED; return; }
  [[ -f $f.aria2 ]] && { echo RESUMABLE; return; }
  if [[ -f $f ]]; then
    if [[ -n ${SIZE:-} && $(fsize "$f") -eq $SIZE ]]; then echo UNVERIFIED; else echo ORPHAN; fi
    return
  fi
  echo PENDING
}

status(){
  local p name st got pct tot=0 done_=0 col lbl
  printf '%s\n' "${BOLD}${C}DeepSeek-V4.1-Flash download${N} ${DIM}$DEST  $(date +%T)${N}"
  for p in "${FILES[@]}"; do
    name=$(lname "$p"); SIZE=""
    [[ -s $META/$name ]] && read -r _ _ SIZE < "$META/$name"
    st=$(state "$name")
    if [[ $st == VERIFIED ]]; then got=${SIZE:-0}; else got=$(ondisk "$DEST/$name"); fi
    pct=0; [[ -n $SIZE && $SIZE -gt 0 ]] && { pct=$(( got*100/SIZE )); (( pct>100 )) && pct=100; tot=$((tot+SIZE)); done_=$((done_+got)); }
    case $st in
      VERIFIED)   col=$G; lbl="✔ verified" ;;
      UNVERIFIED) col=$Y; lbl="● complete, not verified" ;;
      RESUMABLE)  col=$C; lbl="↻ partial, resumable" ;;
      ORPHAN)     col=$R; lbl="✖ partial, NO .aria2 (won't resume)" ;;
      *)          col=$DIM; lbl="· pending" ;;
    esac
    printf ' %s%s%s %3d%%  %-42s %9s / %-9s %s%s%s\n' "$col" "$(bar $pct)" "$N" "$pct" "${name#DeepSeek-V4.1-Flash-1017-}" \
      "$(hum "$got")" "${SIZE:+$(hum "$SIZE")}" "$col" "$lbl" "$N"
  done
  if (( tot>0 )); then printf '%s\n' "${BOLD}TOTAL $(( done_*100/tot ))%  $(hum $done_) / $(hum $tot)${N}"; fi
}

download(){
  local name=$1 rpath=$2 rc
  while true; do
    aria2c -c -d "$DEST" -o "$name" "$BASE/resolve/main/$rpath" \
      --allow-overwrite=false --auto-file-renaming=false --file-allocation=none \
      --max-tries=0 --retry-wait=10 --timeout=60 --connect-timeout=20 \
      -x8 -s8 -k64M --auto-save-interval=10 --summary-interval=0 \
      --console-log-level=warn --human-readable=true --download-result=hide
    rc=$?
    (( rc==0 )) && return 0
    case $rc in
      13) echo "${R}File exists but its .aria2 control file is missing, so aria2 refuses to overwrite it.${N}"
          echo "${R}Delete $DEST/$name to restart that file, then rerun.${N}"; return $rc ;;
      3|24) echo "${R}Permanent error (aria2 exit $rc: 404/auth). Check the URL.${N}"; return $rc ;;
    esac
    echo "${Y}aria2 exit $rc, retrying in 15s (resumes from .aria2)…${N}"; sleep 15
  done
}

verify(){
  local name=$1 exp=$2 f="$DEST/$1" got
  echo "${C}Verifying sha256 of $name …${N}"
  if command -v pv >/dev/null; then
    got=$(pv -pterb -s "$(fsize "$f")" "$f" | "${SHA256[@]}" | awk '{print $1}')
  else
    echo "${DIM}(install pv for a progress bar)${N}"
    got=$("${SHA256[@]}" "$f" | awk '{print $1}')
  fi
  if [[ $got == "$exp" ]]; then
    touch "$META/$name.ok"; echo "${G}✔ sha256 OK${N}"; return 0
  fi
  echo "${R}✖ sha256 MISMATCH${N} expected $exp got $got"
  mv "$f" "$f.BAD"; rm -f "$f.aria2"
  echo "${R}Moved to $name.BAD. Rerun to redownload.${N}"; return 1
}

run(){
  local p name
  echo "${DIM}Fetching expected hashes/sizes from HF…${N}"
  for p in "${FILES[@]}"; do
    name=$(lname "$p")
    fetch_meta "$name" "$p" || { echo "${R}Cannot get metadata for $p (404?)${N}"; exit 1; }
  done
  for p in "${FILES[@]}"; do
    name=$(lname "$p"); fetch_meta "$name" "$p"
    echo; status; echo
    [[ -f $META/$name.ok ]] && continue
    echo "${BOLD}${C}▶ $name${N}"
    if [[ ! -f $DEST/$name || -f $DEST/$name.aria2 ]]; then
      download "$name" "$RPATH" || exit 1
    fi
    if [[ $(fsize "$DEST/$name") -ne $SIZE ]]; then
      echo "${R}Size mismatch: $(fsize "$DEST/$name") vs $SIZE. Rerun.${N}"; exit 1
    fi
    verify "$name" "$SHA" || exit 1
  done
  echo; status; echo "${BOLD}${G}All files downloaded and verified.${N}"
}

trap 'echo; echo "${Y}Interrupted. Rerun ~/dsk-dl.sh to resume.${N}"; exit 130' INT TERM
case "${1:-run}" in
  status) status ;;
  run)    run ;;
  *) echo "usage: $0 [run|status]" ;;
esac
