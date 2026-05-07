#!/bin/sh

middleware=pocolibs
gz_world=/shared-workspace/gazebo/worlds/hexa-ua-world.world

# Genom3 components to run
components="
  nhfc
  pom
  optitrack
  rotorcraft
"

# list of process ids to clean, populated after each spawn
pids=

# cleanup, called after ctrl-C
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
    *) echo "invalid middleware: $middleware";
esac

# optionally run a genomix server for remote control
genomixd & pids="$pids $!"

# spawn required components
for c in $components; do
    $c-$middleware & pids="$pids $!"
done

if [ ! -f $gz_world ]; then
    echo "Cannot find world file: $gz_world"
    exit 1
fi

# start gazebo
gz sim $gz_world & pids="$pids $!"

# wait for ctrl-C or any background process failure
trap atexit CHLD
wait
