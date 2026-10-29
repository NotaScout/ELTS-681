
packaging documentation
https://docs.ros.org/en/eloquent/Tutorials/Writing-A-Simple-Py-Publisher-And-Subscriber.html#create-a-package

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
1
`. install/setup.bash`
2
`ros2 run image_publisher run`

# other stuff that can go wrong

_ARRAY_API not found: 
`
pip uninstall numpy
pip install "numpy<2"
`

