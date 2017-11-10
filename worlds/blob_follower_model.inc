define intersection model
(
  pose [0 0 0]

  # body color
  # determine how the model appears in various sensors

  obstacle_return 0
  laser_return 0
  ranger_return 0
  blobfinder_return 1
  fiducial_return 0
  gripper_return 0
  audio_return 0

  # GUI properties
  gui_nose 0
  gui_grid 0
  gui_boundary 0


  # These corresponds to the maximum x y and z of the table in meters.
  size [1 1 0.2]

  # Top most piece of the table main surface
  block(
  points 4
  point[3] [  1.000  -1.000 ]
  point[2] [ -1.000  -1.000 ]
  point[1] [ -1.000   1.000 ]
  point[0] [  1.000   1.000 ]
  z [ 0.979 1 ]
  )
  )