while true
do
    ps -ef | grep "actionRL1" | grep -v "grep"
    if[ "$?" -eq 1 ]
        then
        ./CFR_action
        echo "process has been restarted"
    else
        echo "process already started!"
    fi
    sleep 10
done