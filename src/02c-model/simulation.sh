#!/bin/sh

middleware=pocolibs

# only nhfc needed — Python simulator replaces Gazebo, rotorcraft, pom, optitrack
components="
  nhfc
"

pids=

atexit() {
    trap - 0 INT CHLD
    set +e

    kill $pids
    wait
    case $middleware in
        pocolibs) h2 end;;
    esac
    exit 0
}
trap atexit 0 INT
set -e

case $middleware in
    pocolibs) h2 init;;
    ros) roscore & pids="$pids $!";;
    *) echo "invalid middleware: $middleware";;
esac

genomixd & pids="$pids $!"

for c in $components; do
    $c-$middleware & pids="$pids $!"
done

trap atexit CHLD
wait
