#!/bin/sh

middleware=pocolibs
gz_world=/opt/openrobots/share/gazebo/worlds/example.world

components="
  nhfc
  pom
  optitrack
  rotorcraft
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

if [ ! -f $gz_world ]; then
    echo "Cannot find world file: $gz_world"
    exit 1
fi

gz sim $gz_world & pids="$pids $!"

trap atexit CHLD
wait
