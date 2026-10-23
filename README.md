


# use these to refresh (& apply the WIFI config)
`sudo netplan --debug try
sudo netplan apply`

# build packages (activate venv)
`
colcon build
`
# local PC
`
sudo apt install ros-humble-image-transport-plugins`

# run image publisher 

# pi side

`. install/setup.bash`

`ros2 run image_publisher run`

