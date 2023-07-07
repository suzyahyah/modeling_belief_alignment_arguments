#!/usr/bin/env bash
# Author: Suzanna Sia

function rm_mk(){
  [ -d $1 ] && rm -r $1
   mkdir -p $1
}

get_seeded_random()
{
    seed="$1"
      openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
}
