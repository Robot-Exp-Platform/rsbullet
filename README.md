# Readme

This project is a fork of the [rubullet](https://github.com/neachdainn/rubullet) project. As the original repository has not been updated for over five years, further maintenance, updates, and customization requirements are being carried out here. The readme file of the original warehouse can be found [README_ORIGIN.md](./README_ORIGIN.md).

Compared to the 100 APIs implemented in the original library, we have additionally implemented all the APIs in pybullet 3.2.7. This enables you to smoothly migrate from pybullet to rsbullet, simply by changing the API names to snake_case. you can read API documentation in the [RSBULLET_API_REFERENCE.md](./RSBULLET_API_REFERENCE.md).

## Overview

### `PhysicsClient` : do everything `PyBullet` can do

In **Rsbullet** you can use the `PhysicsClient` to get features similar to those in PyBullet:

* Create a PhysicsClient in Direct, Gui or other modes
* Load models from URDF, SDF, MuJoCo or Bullet files
* Create own models within the simulation
* Control robots in position, velocity or torque mode
* Calculate inverse dynamics, inverse kinematics, jacobians and mass matrices
* Render camera images
* Read information about links and joints
* Change dynamics of a body
* Create GUI sliders, buttons or put debugging text or lines in the simulation
* Get keyboard and mouse events
* Create and manage constraints
* Logging
* Saving states and loading states
* Set physics engine parameters
* Collision Detection Queries
* Deformables and Cloth
* Everything MultiDOF related
* Virtual Reality
* Plugins

We have prepared a series of examples which can be referred to [examples](./rsbullet/examples).Some of the examples are exactly the same as those in pubullet, which can serve as a standard reference for your migration.

### `Rsbullet` : more features. User-friendly, Simple-interface， Abstraction and More Rustly

Furthermore, we have provided `RsBullet` to achieve functions that `Pybullet` does not have, in order to fully adapt to the features and convenience brought by Rust.

A examples of `Rsbullet` features:

```rust
use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use rsbullet::{Mode, RsBullet};

fn main() -> Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;
    physics
        .set_additional_search_path("E:\\yixing\\code\\Robot-Exp\\drives\\asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;

    let mut robot_1 = physics
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.2, 0.0])
        .base_fixed(true)
        .load()?;

    // a s-curve motion
    robot_1 = robot_1
        .with_joint_vel([5.; 6])
        .with_joint_acc([2.; 6]);
    Motion::move_to::<JointSpace<6>>(&mut robot_1, [0.; 6])?;
    
    loop {
        physics.step()?;
        sleep(Duration::from_secs_f64(0.01));
    }
}
```

In `Pybullet`, to control a robot, you need to get the robot's unique ID first, and then call various functions with the ID as a parameter. In `Rsbullet`, you can directly create a robot object through the `robot_builder` method of `Rsbullet`, and then call the robot's methods in `robot_behavior` to control it. The robot object will automatically manage its own ID internally, making it more convenient to use.

Enjoy it!
