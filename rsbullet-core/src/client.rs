use std::{
    borrow::Cow,
    convert::{TryFrom, TryInto},
    env,
    ffi::{CStr, CString},
    mem::MaybeUninit,
    os::raw::c_char,
    path::{Path, PathBuf},
    ptr, slice,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use nalgebra as na;
use robot_behavior::utils::*;
use rsbullet_sys as ffi;

use crate::error::{BulletError, BulletResult};
use crate::mode::Mode;
use crate::types::*;

// Mirror selected PyBullet limits for reference/statistics
pub const B3_MAX_NUM_END_EFFECTORS: usize = 128;
pub const MAX_PHYSICS_CLIENTS: usize = 1024;

pub struct PhysicsClient {
    handle: ffi::b3PhysicsClientHandle,
    mode: Mode,
}

// Per-process GUI guard: 0 = free, 1 = occupied
static GUI_GUARD: AtomicUsize = AtomicUsize::new(0);
// Process-wide counters to mirror PyBullet statistics
static TOTAL_CLIENTS: AtomicUsize = AtomicUsize::new(0);
static GUI_CLIENTS: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct GeometryScratch {
    cstrings: Vec<CString>,
    float64_buffers: Vec<Vec<f64>>,
    float32_buffers: Vec<Vec<f32>>,
    int32_buffers: Vec<Vec<i32>>,
}

impl GeometryScratch {
    fn push_c_string(&mut self, value: &str) -> BulletResult<*const c_char> {
        let c_string = CString::new(value)?;
        self.cstrings.push(c_string);
        Ok(self.cstrings.last().unwrap().as_ptr())
    }

    fn push_f64_buffer(&mut self, data: Vec<f64>) -> *const f64 {
        self.float64_buffers.push(data);
        self.float64_buffers.last().unwrap().as_ptr()
    }

    #[allow(dead_code)]
    fn push_f64_buffer_mut(&mut self, data: Vec<f64>) -> *mut f64 {
        self.float64_buffers.push(data);
        self.float64_buffers.last_mut().unwrap().as_mut_ptr()
    }

    fn push_f32_buffer_mut(&mut self, data: Vec<f32>) -> *mut f32 {
        self.float32_buffers.push(data);
        self.float32_buffers.last_mut().unwrap().as_mut_ptr()
    }

    fn push_i32_buffer(&mut self, data: Vec<i32>) -> *const i32 {
        self.int32_buffers.push(data);
        self.int32_buffers.last().unwrap().as_ptr()
    }
}

/// ! =====================================================================================================================================
/// ### Connection & Session
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | connect | **Implemented** | `PhysicsClient::connect` covers all modes in scope |
/// | disconnect | **Implemented** | `PhysicsClient::disconnectPhysicsServer` + `Drop` |
/// | getConnectionInfo | **Implemented** | Requires client/server introspection |
/// | isConnected | **Implemented** | Uses `b3CanSubmitCommand` |
impl PhysicsClient {
    /// Connect to an existing physics server (using shared memory by default).  
    /// - connect(method, `key=SHARED_MEMORY_KEY`, options='')
    /// - connect(method, hostname='localhost', port=1234, options='')  
    pub fn connect(mode: Mode) -> BulletResult<Self> {
        // Allow only a single GUI instance per process

        let mut reserved_gui = false;
        if let Mode::Gui | Mode::GuiServer = mode {
            // Atomically reserve if free; otherwise fail
            if GUI_GUARD
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                    if v == 0 { Some(1) } else { None }
                })
                .is_ok()
            {
                reserved_gui = true;
            } else {
                return Err(BulletError::ServerUnavailable(
                    "Only one GUI instance is allowed per process",
                ));
            }
        }

        let handle = match mode {
            Mode::Direct => unsafe { ffi::b3ConnectPhysicsDirect() },
            Mode::Gui => {
                if cfg!(target_os = "macos") {
                    unsafe {
                        ffi::b3CreateInProcessPhysicsServerAndConnectMainThreadSharedMemory(
                            0,
                            ptr::null_mut(),
                        )
                    }
                } else {
                    unsafe {
                        ffi::b3CreateInProcessPhysicsServerAndConnectSharedMemory(
                            0,
                            ptr::null_mut(),
                        )
                    }
                }
            }
            Mode::GuiMainThread => unsafe {
                ffi::b3CreateInProcessPhysicsServerAndConnectMainThreadSharedMemory(
                    0,
                    ptr::null_mut(),
                )
            },
            Mode::GuiServer => {
                if cfg!(target_os = "macos") {
                    unsafe {
                        ffi::b3CreateInProcessPhysicsServerAndConnectMainThreadSharedMemory(
                            0,
                            ptr::null_mut(),
                        )
                    }
                } else {
                    unsafe {
                        ffi::b3CreateInProcessPhysicsServerAndConnectSharedMemory(
                            0,
                            ptr::null_mut(),
                        )
                    }
                }
            }
            Mode::Udp { hostname, port } => {
                if cfg!(feature = "enet") {
                    let host_name = CString::new(hostname)?;
                    unsafe {
                        ffi::b3ConnectPhysicsUDP(host_name.as_ptr(), port.unwrap_or(1234) as i32)
                    }
                } else {
                    return Err(BulletError::ServerUnavailable(
                        "UDP is not enabled in this pybullet build",
                    ));
                }
            }
            Mode::Tcp { hostname, port } => {
                if cfg!(feature = "clsocket") {
                    let host_name = CString::new(hostname)?;
                    unsafe {
                        ffi::b3ConnectPhysicsTCP(host_name.as_ptr(), port.unwrap_or(6667) as i32)
                    }
                } else {
                    return Err(BulletError::ServerUnavailable(
                        "TCP is not enabled in this pybullet build",
                    ));
                }
            }
            Mode::GraphicsServerMainThread { port } => unsafe {
                ffi::b3CreateInProcessGraphicsServerAndConnectMainThreadSharedMemory(
                    port.unwrap_or(6667) as i32,
                )
            },
            Mode::GraphicsServer { port } => {
                if cfg!(target_os = "macos") {
                    unsafe {
                        ffi::b3CreateInProcessGraphicsServerAndConnectMainThreadSharedMemory(
                            port.unwrap_or(6667) as i32,
                        )
                    }
                } else {
                    unsafe {
                        ffi::b3CreateInProcessGraphicsServerAndConnectSharedMemory(
                            port.unwrap_or(6667) as i32,
                        )
                    }
                }
            }
            Mode::GraphicsServerTcp { hostname, port } => {
                if cfg!(feature = "clsocket") {
                    let host_name = CString::new(hostname)?;
                    unsafe {
                        ffi::b3ConnectPhysicsTCP(host_name.as_ptr(), port.unwrap_or(6667) as i32)
                    }
                } else {
                    return Err(BulletError::ServerUnavailable(
                        "TCP is not enabled in this pybullet build",
                    ));
                }
            }
            Mode::SharedMemory => unsafe { ffi::b3ConnectSharedMemory(ffi::SHARED_MEMORY_KEY) },
            Mode::SharedMemoryServer => unsafe {
                ffi::b3CreateInProcessPhysicsServerFromExistingExampleBrowserAndConnect3(
                    ptr::null_mut(),
                    ffi::SHARED_MEMORY_KEY,
                )
            },
            Mode::SharedMemoryGui | Mode::GraphicsClient => unsafe {
                ffi::b3CreateInProcessPhysicsServerFromExistingExampleBrowserAndConnect4(
                    ptr::null_mut(),
                    ffi::SHARED_MEMORY_KEY,
                )
            },

            #[cfg(feature = "dart")]
            Mode::Dart => unsafe { ffi::b3ConnectPhysicsDART() },
            #[cfg(feature = "physx")]
            Mode::PhysX => unsafe { ffi::b3ConnectPhysicsPhysX(0, ptr::null_mut()) },
            #[cfg(feature = "mujoco")]
            Mode::MuJoCo => unsafe { ffi::b3ConnectPhysicsMuJoCo() },
            #[cfg(feature = "grpc")]
            Mode::Grpc { hostname, port } => {
                let host_name = CString::new(hostname)?;
                unsafe { ffi::b3ConnectPhysicsGRPC(host_name.as_ptr(), port.unwrap_or(-1)) }
            }
        }
        .ok_or(BulletError::NullPointer(
            "Bullet returned a null physics client handle",
        ))?;

        let mut client = PhysicsClient { handle, mode };
        client.ensure_can_submit()?;

        unsafe {
            let command = ffi::b3InitSyncBodyInfoCommand(client.handle);
            client.submit_simple_command(
                command,
                ffi::EnumSharedMemoryServerStatus::CMD_SYNC_BODY_INFO_COMPLETED,
            )?;

            let command = ffi::b3InitSyncUserDataCommand(client.handle);
            client.submit_simple_command(
                command,
                ffi::EnumSharedMemoryServerStatus::CMD_SYNC_USER_DATA_COMPLETED,
            )?;
        }

        // Successful connect: update counters
        TOTAL_CLIENTS.fetch_add(1, Ordering::SeqCst);
        if reserved_gui {
            GUI_CLIENTS.fetch_add(1, Ordering::SeqCst);
        }

        Ok(client)
    }

    pub fn get_connection_info(&mut self) -> BulletResult<String> {
        let is_connected = self.is_connected();
        let mode = &self.mode;
        Ok(format!(
            "{{\"isConnected\": {is_connected}, \"mode\": \"{mode:?}\"}}"
        ))
    }

    /// "Disconnect from the physics server."
    pub fn disconnect(self) {
        drop(self);
    }

    pub fn is_connected(&mut self) -> bool {
        self.can_submit_command()
    }

    /// Returns whether or not this client can submit a command.
    pub(crate) fn can_submit_command(&mut self) -> bool {
        unsafe { ffi::b3CanSubmitCommand(self.handle) != 0 }
    }
}

/// ! =====================================================================================================================================
/// ### Simulation Parameters
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | resetSimulation | **Implemented** | Core reset command |
/// | stepSimulation | **Implemented** | Core stepping command |
/// | performCollisionDetection | **Implemented** | Extra collision command |
/// | setGravity | **Implemented** | Physics param command |
/// | setTimeStep | **Implemented** | Physics param command |
/// | setDefaultContactERP | **Implemented** | Physics param command |
/// | setRealTimeSimulation | **Implemented** | Physics param command |
/// | setPhysicsEngineParameter | **Implemented** | Bulk physics-param configuration |
/// | getPhysicsEngineParameters | **Implemented** | Query mirror of above |
/// | setInternalSimFlags | **Implemented** | Expert-only flagging |
impl PhysicsClient {
    /// `reset_simulation` will remove all objects from the world and reset the world to initial conditions.
    pub fn reset_simulation(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitResetSimulationCommand(self.handle) };
        unsafe { ffi::b3InitResetSimulationSetFlags(command, 0) };
        let _status_handle =
            unsafe { ffi::b3SubmitClientCommandAndWaitStatus(self.handle, command) };
        Ok(())
    }

    /// Performs all the actions in a single forward dynamics simulation step such as collision
    /// detection, constraint solving, and integration.
    /// TODO: Return analytics data?
    pub fn step_simulation(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitStepSimulationCommand(self.handle) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_STEP_FORWARD_SIMULATION_COMPLETED,
        )?;
        Ok(())
    }

    pub fn perform_collision_detection(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPerformCollisionDetectionCommand(self.handle) };
        let _ = self.submit_command(command)?;
        Ok(())
    }

    /// Sets the default gravity force for all objects.
    ///
    /// By default, there is no gravitational force enabled.
    ///
    /// # Arguments
    /// * `gravity` - a gravity vector. Can be a \[f64;3\]-array or anything else that can be
    ///   converted into [f64; 3].
    pub fn set_gravity(&mut self, grav: impl Into<[f64; 3]>) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let grav = grav.into();
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        unsafe {
            ffi::b3PhysicsParamSetGravity(command, grav[0], grav[1], grav[2]);
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    /// Warning: in many cases it is best to leave the timeStep to default, which is 240Hz.
    /// Several parameters are tuned with this value in mind. For example the number of solver
    /// iterations and the error reduction parameters (erp) for contact, friction and non-contact
    /// joints are related to the time step. If you change the time step, you may need to re-tune
    /// those values accordingly, especially the erp values.
    ///
    /// You can set the physics engine timestep that is used when calling
    /// [`step_simulation`](`Self::step_simulation()`).
    /// It is best to only call this method at the start of a simulation.
    /// Don't change this time step regularly.
    pub fn set_time_step(&mut self, time_step: Duration) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        unsafe {
            ffi::b3PhysicsParamSetTimeStep(command, time_step.as_secs_f64());
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    /// Update the global default contact ERP (error reduction parameter).
    pub fn set_default_contact_erp(&mut self, default_contact_erp: f64) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        unsafe {
            ffi::b3PhysicsParamSetDefaultContactERP(command, default_contact_erp);
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    /// By default, the physics server will not step the simulation, unless you explicitly send a
    /// [`step_simulation`](`Self::step_simulation()`) command.
    /// This way you can maintain control determinism of the simulation
    /// It is possible to run the simulation in real-time by letting the physics server
    /// automatically step the simulation according to its real-time-clock (RTC) using the
    /// `set_real_time_simulation` command. If you enable the real-time simulation,
    /// you don't need to call [`step_simulation`](`Self::step_simulation()`).
    ///
    /// Note that `set_real_time_simulation` has no effect in
    /// [`Direct mode`](`crate::mode::Mode::Direct`) :
    /// in [`Direct mode`](`crate::mode::Mode::Direct`) mode the physics
    /// server and client happen in the same thread and you trigger every command.
    /// In [`Gui mode`](`crate::mode::Mode::Gui`) and in Virtual Reality mode, and TCP/UDP mode,
    /// the physics server runs in a separate thread from the client (`RuBullet`),
    /// and `set_real_time_simulation` allows the physics server thread
    /// to add additional calls to  [`step_simulation`](`Self::step_simulation()`).
    ///
    /// # Arguments
    /// * `enable_real_time_simulation` - activates or deactivates real-time simulation
    pub fn set_real_time_simulation(&mut self, enable: bool) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        unsafe {
            ffi::b3PhysicsParamSetRealTimeSimulation(command, enable as i32);
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    /// Apply a batch of physics engine parameters, mirroring `PyBullet`'s `setPhysicsEngineParameter`.
    pub fn set_physics_engine_parameter(
        &mut self,
        update: &PhysicsEngineParametersUpdate,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        Self::apply_physics_engine_update(command, update);
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    /// Query the current physics simulation parameters from the server.
    pub fn get_physics_engine_parameters(&mut self) -> BulletResult<PhysicsEngineParameters> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRequestPhysicsParamCommand(self.handle) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_PHYSICS_SIMULATION_PARAMETERS_COMPLETED,
        )?;

        let mut raw = MaybeUninit::<ffi::b3PhysicsSimulationParameters>::uninit();
        let result =
            unsafe { ffi::b3GetStatusPhysicsSimulationParameters(status.handle, raw.as_mut_ptr()) };
        if result == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to fetch physics simulation parameters",
                code: result,
            });
        }
        unsafe { raw.assume_init() }.try_into()
    }

    /// Set internal simulation flags (expert-level API).
    pub fn set_internal_sim_flags(&mut self, flags: i32) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitPhysicsParamCommand(self.handle) };
        unsafe {
            ffi::b3PhysicsParamSetInternalSimFlags(command, flags);
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }
}

// ! =====================================================================================================================================
// ### World Authoring & Persistence
//
// | API | Status | Notes |
// | --- | --- | --- |
// | loadURDF | **Implemented** | Supports position/orientation/options |
// | loadSDF | **Implemented** | Returns body list |
// | loadSoftBody | Optional | Requires soft-body build support |
// | createSoftBodyAnchor | Optional | Soft-body specific |
// | loadBullet | **Implemented** | World snapshot load |
// | saveBullet | **Implemented** | World snapshot save |
// | restoreState | **Implemented** | In-memory state restore |
// | saveState | **Implemented** | In-memory state save |
// | removeState | **Implemented** | Pair with saveState |
// | loadMJCF | **Implemented** | Returns body list |
// | saveWorld | **Implemented** | Script export helper |
// | setAdditionalSearchPath | **Implemented** | Search path registry |
// | vhacd | Optional | Requires VHACD build flag |

impl PhysicsClient {
    /// Sends a command to the physics server to load a physics model from a Unified Robot
    /// Description Format (URDF) model.
    ///
    /// # Arguments
    /// * `file` - a relative or absolute path to the URDF file on the file system of the physics server
    /// * `options` - use additional options like `global_scaling` and `use_maximal_coordinates` for
    ///   loading the URDF-file. See [`UrdfOptions`](`crate::types::UrdfOptions`).
    /// # Example
    /// ```rust
    /// use rsbullet_core::*;
    /// fn main() -> BulletResult<&mut Self> {
    ///     let mut physics_client = PhysicsClient::connect(Mode::Direct)?;
    ///     physics_client.set_additional_search_path(PhysicsClient::bullet_data_path())?;
    ///     let cube = physics_client.load_urdf("cube.urdf", None)?;
    ///     assert_eq!("baseLink", physics_client.get_body_info(cube)?.base_name);
    ///     for i in 0..10 {
    ///         let _cube = physics_client.load_urdf(
    ///             "cube.urdf",
    ///             None,
    ///         )?;
    ///     }
    ///     assert_eq!(11, physics_client.get_num_bodies());
    ///     Ok(())
    /// }
    /// ```
    pub fn load_urdf(
        &mut self,
        file: impl AsRef<Path>,
        options: Option<impl Into<UrdfOptions>>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3LoadUrdfCommandInit(self.handle, file_c.as_ptr()) };

        if let Some(options) = options {
            let options = options.into();

            if let Some(flags) = options.flags {
                unsafe {
                    ffi::b3LoadUrdfCommandSetFlags(command, flags.bits());
                }
            }

            if let Some(base) = options.base.as_ref() {
                let ([x, y, z], [ox, oy, oz, ow]) = isometry_to_raw_parts(base);
                unsafe {
                    ffi::b3LoadUrdfCommandSetStartPosition(command, x, y, z);
                    ffi::b3LoadUrdfCommandSetStartOrientation(command, ox, oy, oz, ow);
                }
            }

            if let Some(use_maximal) = options.use_maximal_coordinates {
                unsafe {
                    ffi::b3LoadUrdfCommandSetUseMultiBody(command, (!use_maximal) as i32);
                }
            }

            if options.use_fixed_base {
                unsafe {
                    ffi::b3LoadUrdfCommandSetUseFixedBase(command, 1);
                }
            }

            if let Some(global_scaling) = options.global_scaling {
                unsafe {
                    ffi::b3LoadUrdfCommandSetGlobalScaling(command, global_scaling);
                }
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_URDF_LOADING_COMPLETED,
        )?;

        let body_id = unsafe { ffi::b3GetStatusBodyIndex(status.handle) };
        Ok(body_id)
    }

    /// Sends a command to the physics server to load a physics model from
    /// a Simulation Description Format (SDF) model.
    /// # Arguments
    /// * `file` - a relative or absolute path to the SDF file on the file system of the physics server.
    /// * `options` -  use additional options like `global_scaling` and `use_maximal_coordinates` for
    ///   loading the SDF-file. See [`SdfOptions`](`crate::types::SdfOptions`).
    ///    
    /// # Return
    /// Returns a list of unique body id's
    ///
    /// # Example
    /// ```rust
    /// use rsbullet_core::*;
    /// fn main() -> BulletResult<&mut Self> {
    ///     let mut physics_client = PhysicsClient::connect(Mode::Direct)?;
    ///     physics_client.set_additional_search_path(PhysicsClient::bullet_data_path())?;
    ///     let cubes = physics_client.load_sdf("two_cubes.sdf", None)?;
    ///     assert_eq!(3, cubes.len()); // 2 cubes + 1 plane
    ///     Ok(())
    /// }
    /// ```
    pub fn load_sdf(
        &mut self,
        file: impl AsRef<Path>,
        options: impl Into<Option<SdfOptions>>,
    ) -> BulletResult<Vec<i32>> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3LoadSdfCommandInit(self.handle, file_c.as_ptr()) };

        let options = options.into().unwrap_or_default();

        if let Some(use_maximal) = options.use_maximal_coordinates {
            unsafe {
                ffi::b3LoadSdfCommandSetUseMultiBody(command, (!use_maximal) as i32);
            }
        }

        if let Some(global_scaling) = options.global_scaling {
            unsafe {
                ffi::b3LoadSdfCommandSetUseGlobalScaling(command, global_scaling);
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_SDF_LOADING_COMPLETED,
        )?;

        Self::collect_body_indices(status.handle)
    }

    /// Sends a command to the physics server to load a physics model from
    /// a `MuJoCo` model.
    /// # Arguments
    /// * `file` - a relative or absolute path to the `MuJoCo` file on the file system of the physics server.
    /// * `flags` -  Flags for loading the model. Set to None if you do not whish to provide any.
    ///
    /// # Return
    /// Returns a list of unique body id's
    ///
    /// # Example
    /// ```rust
    /// use rsbullet_core::*;
    /// fn main() -> BulletResult<&mut Self> {
    ///     let mut physics_client = PhysicsClient::connect(Mode::Direct)?;
    ///     physics_client.set_additional_search_path(PhysicsClient::bullet_data_path())?;
    ///     let stadium = physics_client.load_mjcf("mjcf/hello_mjcf.xml", None)?;
    ///     assert_eq!(2, stadium.len()); // 1 cube + 1 plane
    ///
    ///     let plane = physics_client.load_mjcf("mjcf/ground_plane.xml", LoadModelFlags::URDF_ENABLE_CACHED_GRAPHICS_SHAPES)?;
    ///     assert_eq!(1, plane.len());
    ///     Ok(())
    /// }
    /// ```
    pub fn load_mjcf(
        &mut self,
        file: impl AsRef<Path>,
        options: impl Into<Option<MjcfOptions>>,
    ) -> BulletResult<Vec<i32>> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3LoadMJCFCommandInit(self.handle, file_c.as_ptr()) };

        let options = options.into().unwrap_or_default();

        if let Some(flags) = options.flags {
            unsafe {
                ffi::b3LoadMJCFCommandSetFlags(command, flags.bits());
            }
        }

        if let Some(use_multi_body) = options.use_multi_body {
            unsafe {
                ffi::b3LoadMJCFCommandSetUseMultiBody(command, use_multi_body as i32);
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_MJCF_LOADING_COMPLETED,
        )?;

        Self::collect_body_indices(status.handle)
    }

    /// Loads Bodies from a `.bullet` file. These can be created with [`save_bullet`](`Self::save_bullet`).
    ///
    /// Returns a list of `BodyId`'s.
    /// # Arguments
    /// * `bullet_filename` - location of the `.bullet`
    /// # Example
    /// ```rust
    ///# use rsbullet::*;
    ///#
    ///# fn main() -> BulletResult<&mut Self> {
    ///#     let mut physics_client = PhysicsClient::connect(Mode::Direct)?;
    ///#     physics_client.set_additional_search_path(PhysicsClient::bullet_data_path())?;
    ///     let points = physics_client.load_bullet("spider.bullet")?;
    ///#     Ok(())
    ///# }
    /// ```
    /// See also `save_and_restore.rs` example.
    pub fn load_bullet(&mut self, file: impl AsRef<Path>) -> BulletResult<Vec<i32>> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3LoadBulletCommandInit(self.handle, file_c.as_ptr()) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_BULLET_LOADING_COMPLETED,
        )?;

        Self::collect_body_indices(status.handle)
    }
    /// Saves all bodies and the current state into a `.bullet` file which can then be read by
    /// [`load_bullet`](`Self::load_bullet`) or [`restore_state_from_file`](`Self::restore_state_from_file`).
    /// See also `save_and_restore.rs` example.
    pub fn save_bullet(&mut self, file: impl AsRef<Path>) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3SaveBulletCommandInit(self.handle, file_c.as_ptr()) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_BULLET_SAVING_COMPLETED,
        )?;
        Ok(())
    }

    pub fn save_world(&mut self, file: impl AsRef<Path>) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file.as_ref())?;
        let command = unsafe { ffi::b3SaveWorldCommandInit(self.handle, file_c.as_ptr()) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_SAVE_WORLD_COMPLETED,
        )?;
        Ok(())
    }

    /// restores a state from memory using a state id which was created with [`save_state`](`Self::save_state`).
    /// See `save_and_restore.rs` example.
    pub fn restore_state(
        &mut self,
        state_id: Option<i32>,
        file: Option<impl AsRef<Path>>,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3LoadStateCommandInit(self.handle) };

        if let Some(state_id) = state_id {
            unsafe {
                ffi::b3LoadStateSetStateId(command, state_id);
            }
        }

        if let Some(ref file) = file {
            let file_c = Self::path_to_cstring(file.as_ref())?;
            unsafe { ffi::b3LoadStateSetFileName(command, file_c.as_ptr()) };
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_RESTORE_STATE_COMPLETED,
        )?;
        Ok(())
    }
    /// Saves the current state in memory and returns a `StateId` which can be used by [`restore_state`](`Self::restore_state`)
    /// to restore this state.  Use [`save_bullet`](`Self::save_bullet`) if you want to save a state
    /// to a file.
    /// See `save_and_restore.rs` example.
    pub fn save_state(&mut self) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3SaveStateCommandInit(self.handle) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_SAVE_STATE_COMPLETED,
        )?;

        let state_id = unsafe { ffi::b3GetStatusGetStateId(status.handle) };
        Ok(state_id)
    }
    /// Removes a state from memory.
    pub fn remove_state(&mut self, state_id: i32) -> BulletResult<()> {
        if state_id < 0 {
            return Ok(());
        }
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRemoveStateCommand(self.handle, state_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REMOVE_STATE_COMPLETED,
        )?;
        Ok(())
    }

    pub fn set_default_search_path(&mut self) -> BulletResult<&mut Self> {
        self.set_additional_search_path(Self::bullet_data_path())
    }

    pub fn set_additional_search_path(
        &mut self,
        path: impl AsRef<Path>,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let path_c = Self::path_to_cstring(path.as_ref())?;
        let command = unsafe { ffi::b3SetAdditionalSearchPath(self.handle, path_c.as_ptr()) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn bullet_data_path() -> PathBuf {
        {
            #[cfg(target_os = "windows")]
            {
                env::var_os("LOCALAPPDATA")
                    .or_else(|| env::var_os("USERPROFILE"))
                    .map(PathBuf::from)
            }

            #[cfg(target_os = "macos")]
            {
                use std::path::PathBuf;
                env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join("Library").join("Application Support"))
            }

            #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
            {
                use std::path::PathBuf;
                env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".local").join("share"))
            }
        }
        .map(|path| path.join("bullet"))
        .unwrap()
    }
}

/// ! =====================================================================================================================================
/// ### Asset Creation & Mutation
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | createCollisionShape | **Implemented** | Supports primitive/mesh geometry |
/// | createCollisionShapeArray | **Implemented** | Compound shape builder |
/// | removeCollisionShape | **Implemented** | Clean-up helper |
/// | getMeshData | Pending | Mesh inspection |
/// | getTetraMeshData | Pending | Soft-body mesh |
/// | resetMeshData | Optional | Deformable specific |
/// | createVisualShape | **Implemented** | Visual geometry authoring |
/// | createVisualShapeArray | **Implemented** | Bulk visual authoring |
/// | createMultiBody | **Implemented** | Procedural multibody build |
/// | createConstraint | **Implemented** | Constraint authoring |
/// | changeConstraint | **Implemented** | Constraint mutation |
/// | removeConstraint | **Implemented** | Constraint teardown |
/// | enableJointForceTorqueSensor | **Implemented** | Sensor toggle |
/// | removeBody | **Implemented** | Body teardown |
/// | getNumConstraints | **Implemented** | Constraint enumeration |
/// | getConstraintInfo | **Implemented** | Constraint query |
/// | getConstraintState | **Implemented** | Constraint forces |
/// | getConstraintUniqueId | **Implemented** | Constraint enumeration |
/// | changeVisualShape | **Implemented** | Visual mutation |
/// | resetVisualShapeData | Pending | Legacy alias |
/// | loadTexture | **Implemented** | Visual assets |
/// | changeTexture | **Implemented** | Visual assets |
impl PhysicsClient {
    /// You can create a collision shape in a similar way to creating a visual shape. If you have
    /// both you can use them to create objects in `RuBullet`.
    /// # Arguments
    /// * `shape` - A geometric body from which to create the shape
    /// * `frame_offset` - offset of the shape with respect to the link frame. Default is no offset.
    ///
    /// # Return
    /// Returns a unique [`CollisionId`](crate::CollisionId) which can then be used to create a body.
    /// # See also
    /// * [`create_visual_shape`](`Self::create_visual_shape`)
    /// * [`create_multi_body`](`Self::create_multi_body`)
    pub fn create_collision_shape(
        &mut self,
        geometry: &CollisionGeometry<'_>,
        options: Option<impl Into<CollisionShapeOptions>>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let mut scratch = GeometryScratch::default();
        let command = unsafe { ffi::b3CreateCollisionShapeCommandInit(self.handle) };
        let shape_index = self.add_collision_geometry(command, geometry, &mut scratch)?;

        let CollisionShapeOptions { transform, flags } =
            options.map_or(CollisionShapeOptions::default(), Into::into);

        if let Some(flags) = flags {
            unsafe { ffi::b3CreateCollisionSetFlag(command, shape_index, flags) };
        }

        let (position, orientation) = isometry_to_raw_parts(&transform);
        unsafe {
            ffi::b3CreateCollisionShapeSetChildTransform(
                command,
                shape_index,
                position.as_ptr(),
                orientation.as_ptr(),
            );
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CREATE_COLLISION_SHAPE_COMPLETED,
        )?;
        let shape_id = unsafe { ffi::b3GetStatusCollisionShapeUniqueId(status.handle) };
        if shape_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create collision shape",
                code: shape_id,
            });
        }
        Ok(shape_id)
    }

    pub fn create_collision_shape_array(
        &mut self,
        options: &'_ [(
            CollisionGeometry<'_>,
            impl Clone + Into<CollisionShapeOptions>,
        )],
    ) -> BulletResult<i32> {
        if options.is_empty() {
            return Err(BulletError::CommandFailed {
                message: "Collision shape array requires at least one entry",
                code: -1,
            });
        }

        self.ensure_can_submit()?;
        let mut scratch = GeometryScratch::default();
        let command = unsafe { ffi::b3CreateCollisionShapeCommandInit(self.handle) };

        for child in options {
            let (geometry, option) = child;
            let CollisionShapeOptions { transform, flags } = option.clone().into();
            let index = self.add_collision_geometry(command, geometry, &mut scratch)?;
            if let Some(flags) = flags {
                unsafe { ffi::b3CreateCollisionSetFlag(command, index, flags) };
            }
            let (position, orientation) = isometry_to_raw_parts(&transform);
            unsafe {
                ffi::b3CreateCollisionShapeSetChildTransform(
                    command,
                    index,
                    position.as_ptr(),
                    orientation.as_ptr(),
                );
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CREATE_COLLISION_SHAPE_COMPLETED,
        )?;
        let shape_id = unsafe { ffi::b3GetStatusCollisionShapeUniqueId(status.handle) };
        if shape_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create collision shape array",
                code: shape_id,
            });
        }
        Ok(shape_id)
    }

    pub fn remove_collision_shape(&mut self, collision_shape_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command =
            unsafe { ffi::b3InitRemoveCollisionShapeCommand(self.handle, collision_shape_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    /// You can create a visual shape in a similar way to creating a collision shape, with some
    /// additional arguments to control the visual appearance, such as diffuse and specular color.
    /// When you use the [`GeometricVisualShape::MeshFile`](`crate::GeometricVisualShape::MeshFile`)
    /// type, you can point to a Wavefront OBJ file, and the
    /// visual shape will parse some parameters from the material file (.mtl) and load a texture.
    /// Note that large textures (above 1024x1024 pixels)
    /// can slow down the loading and run-time performance.
    ///
    /// # Arguments
    /// * `shape` - A geometric body from which to create the shape
    /// * `options` - additional options to specify, like colors. See [`VisualShapeOptions`](crate::VisualShapeOptions)
    ///   for details.
    /// # Return
    /// Returns a unique [`VisualId`](crate::VisualId) which can then be used to create a body.
    /// # See also
    /// * [`create_collision_shape`](`Self::create_collision_shape`)
    /// * [`create_multi_body`](`Self::create_multi_body`)
    pub fn create_visual_shape(
        &mut self,
        geometry: &VisualGeometry<'_>,
        options: Option<impl Into<VisualShapeOptions>>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let mut scratch = GeometryScratch::default();
        let command = unsafe { ffi::b3CreateVisualShapeCommandInit(self.handle) };

        let shape_index = self.add_visual_geometry(command, geometry, &mut scratch)?;

        let options = options.map_or(VisualShapeOptions::default(), Into::into);
        if let Some(flags) = options.flags {
            unsafe { ffi::b3CreateVisualSetFlag(command, shape_index, flags.bits()) };
        }
        let (position, orientation) = isometry_to_raw_parts(&options.transform);
        unsafe {
            ffi::b3CreateVisualShapeSetChildTransform(
                command,
                shape_index,
                position.as_ptr(),
                orientation.as_ptr(),
            );
        }
        unsafe {
            ffi::b3CreateVisualShapeSetRGBAColor(command, shape_index, options.rgba.as_ptr());
        }
        unsafe {
            ffi::b3CreateVisualShapeSetSpecularColor(
                command,
                shape_index,
                options.specular.as_ptr(),
            );
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CREATE_VISUAL_SHAPE_COMPLETED,
        )?;
        let visual_id = unsafe { ffi::b3GetStatusVisualShapeUniqueId(status.handle) };
        if visual_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create visual shape",
                code: visual_id,
            });
        }
        Ok(visual_id)
    }

    pub fn create_visual_shape_array(
        &mut self,
        options: &'_ [(VisualGeometry, Option<VisualShapeOptions>)],
    ) -> BulletResult<i32> {
        if options.is_empty() {
            return Err(BulletError::CommandFailed {
                message: "Visual shape array requires at least one entry",
                code: -1,
            });
        }

        self.ensure_can_submit()?;
        let mut scratch = GeometryScratch::default();
        let command = unsafe { ffi::b3CreateVisualShapeCommandInit(self.handle) };

        for child in options {
            let (geometry, options) = child;

            let shape_index = self.add_visual_geometry(command, geometry, &mut scratch)?;

            let options = options.unwrap_or_default();
            if let Some(flags) = &options.flags {
                unsafe { ffi::b3CreateVisualSetFlag(command, shape_index, flags.bits()) };
            }
            let (position, orientation) = isometry_to_raw_parts(&options.transform);
            unsafe {
                ffi::b3CreateVisualShapeSetChildTransform(
                    command,
                    shape_index,
                    position.as_ptr(),
                    orientation.as_ptr(),
                );
            }
            unsafe {
                ffi::b3CreateVisualShapeSetRGBAColor(command, shape_index, options.rgba.as_ptr());
            }
            unsafe {
                ffi::b3CreateVisualShapeSetSpecularColor(
                    command,
                    shape_index,
                    options.specular.as_ptr(),
                );
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CREATE_VISUAL_SHAPE_COMPLETED,
        )?;
        let visual_id = unsafe { ffi::b3GetStatusVisualShapeUniqueId(status.handle) };
        if visual_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create visual shape array",
                code: visual_id,
            });
        }
        Ok(visual_id)
    }

    pub fn change_visual_shape(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        options: &ChangeVisualShapeOptions,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3InitUpdateVisualShape2(
                self.handle,
                body_unique_id,
                link_index,
                options.shape_index.0,
            )
        };

        if let Some(texture_id) = options.texture_unique_id {
            unsafe { ffi::b3UpdateVisualShapeTexture(command, texture_id) };
        }
        if let Some(rgba) = options.rgba_color {
            unsafe { ffi::b3UpdateVisualShapeRGBAColor(command, rgba.as_ptr()) };
        }
        if let Some(flags) = options.flags {
            unsafe { ffi::b3UpdateVisualShapeFlags(command, flags) };
        }
        if let Some(specular) = options.specular_color {
            unsafe { ffi::b3UpdateVisualShapeSpecularColor(command, specular.as_ptr()) };
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_VISUAL_SHAPE_UPDATE_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn create_multi_body(&mut self, options: &MultiBodyCreateOptions<'_>) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreateMultiBodyCommandInit(self.handle) };

        let (base_position, base_orientation) = isometry_to_raw_parts(&options.base.pose);
        let (inertial_position, inertial_orientation) =
            isometry_to_raw_parts(&options.base.inertial_pose);

        unsafe {
            ffi::b3CreateMultiBodyBase(
                command,
                options.base.mass,
                options.base.collision_shape.0,
                options.base.visual_shape.0,
                base_position.as_ptr(),
                base_orientation.as_ptr(),
                inertial_position.as_ptr(),
                inertial_orientation.as_ptr(),
            );
        }

        for link in options.links {
            let parent_index = match link.parent_index {
                Some(index) => Self::usize_to_i32(index)?,
                None => -1,
            };

            let (link_position, link_orientation) = isometry_to_raw_parts(&link.parent_transform);
            let (link_inertial_position, link_inertial_orientation) =
                isometry_to_raw_parts(&link.inertial_transform);

            unsafe {
                ffi::b3CreateMultiBodyLink(
                    command,
                    link.mass,
                    link.collision_shape.0 as f64,
                    link.visual_shape.0 as f64,
                    link_position.as_ptr(),
                    link_orientation.as_ptr(),
                    link_inertial_position.as_ptr(),
                    link_inertial_orientation.as_ptr(),
                    parent_index,
                    link.joint_type as i32,
                    link.joint_axis.as_ptr(),
                );
            }
        }

        if let Some(flags) = options.flags {
            unsafe { ffi::b3CreateMultiBodySetFlags(command, flags) };
        }
        if options.use_maximal_coordinates {
            unsafe { ffi::b3CreateMultiBodyUseMaximalCoordinates(command) };
        }
        let mut _batch_storage: Option<Vec<f64>> = None;
        if let Some(batch) = options.batch_transforms {
            let mut flattened = Self::flatten_isometries(batch);
            unsafe {
                ffi::b3CreateMultiBodySetBatchPositions(
                    self.handle,
                    command,
                    flattened.as_mut_ptr(),
                    batch.len() as i32,
                );
            }
            _batch_storage = Some(flattened);
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CREATE_MULTI_BODY_COMPLETED,
        )?;
        let body_id = unsafe { ffi::b3GetStatusBodyIndex(status.handle) };
        if body_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create multibody",
                code: body_id,
            });
        }
        Ok(body_id)
    }

    pub fn create_constraint(&mut self, options: &ConstraintCreateOptions) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let parent_link = match options.parent_link {
            Some(index) => Self::usize_to_i32(index)?,
            None => -1,
        };
        let child_link = match options.child_link {
            Some(index) => Self::usize_to_i32(index)?,
            None => -1,
        };
        let mut joint_info = ffi::b3JointInfo {
            m_joint_type: options.joint_type as i32,
            m_joint_axis: options.joint_axis,
            m_joint_max_force: options.max_applied_force,
            ..Default::default()
        };
        Self::write_transform_to_frame(&options.parent_frame, &mut joint_info.m_parent_frame);
        Self::write_transform_to_frame(&options.child_frame, &mut joint_info.m_child_frame);

        let command = unsafe {
            ffi::b3InitCreateUserConstraintCommand(
                self.handle,
                options.parent_body,
                parent_link,
                options.child_body,
                child_link,
                &raw mut joint_info,
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_CONSTRAINT_COMPLETED,
        )?;
        let constraint_id = unsafe { ffi::b3GetStatusUserConstraintUniqueId(status.handle) };
        if constraint_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to create user constraint",
                code: constraint_id,
            });
        }

        let update = ConstraintUpdate {
            max_force: Some(options.max_applied_force),
            gear_ratio: options.gear_ratio,
            gear_aux_link: options.gear_aux_link,
            relative_position_target: options.relative_position_target,
            erp: options.erp,
            ..Default::default()
        };

        if Self::constraint_update_has_changes(&update) {
            self.change_constraint(constraint_id, &update)?;
        }

        Ok(constraint_id)
    }

    pub fn change_constraint(
        &mut self,
        constraint_id: i32,
        update: &ConstraintUpdate,
    ) -> BulletResult<&mut Self> {
        if !Self::constraint_update_has_changes(update) {
            return Ok(self);
        }

        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitChangeUserConstraintCommand(self.handle, constraint_id) };

        if let Some(frame) = update.child_frame.as_ref() {
            let (pivot, orientation) = isometry_to_raw_parts(frame);
            unsafe {
                ffi::b3InitChangeUserConstraintSetPivotInB(command, pivot.as_ptr());
                ffi::b3InitChangeUserConstraintSetFrameInB(command, orientation.as_ptr());
            }
        }
        if let Some(force) = update.max_force {
            unsafe { ffi::b3InitChangeUserConstraintSetMaxForce(command, force) };
        }
        if let Some(ratio) = update.gear_ratio {
            unsafe { ffi::b3InitChangeUserConstraintSetGearRatio(command, ratio) };
        }
        if let Some(aux) = update.gear_aux_link {
            let aux = Self::usize_to_i32(aux)?;
            unsafe { ffi::b3InitChangeUserConstraintSetGearAuxLink(command, aux) };
        }
        if let Some(target) = update.relative_position_target {
            unsafe { ffi::b3InitChangeUserConstraintSetRelativePositionTarget(command, target) };
        }
        if let Some(erp) = update.erp {
            unsafe { ffi::b3InitChangeUserConstraintSetERP(command, erp) };
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CHANGE_USER_CONSTRAINT_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn remove_constraint(&mut self, constraint_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRemoveUserConstraintCommand(self.handle, constraint_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REMOVE_USER_CONSTRAINT_COMPLETED,
        )?;
        Ok(())
    }

    pub fn enable_joint_force_torque_sensor(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
        enable: bool,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreateSensorCommandInit(self.handle, body_unique_id) };
        unsafe {
            ffi::b3CreateSensorEnable6DofJointForceTorqueSensor(
                command,
                joint_index,
                enable.into(),
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn remove_body(&mut self, body_unique_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRemoveBodyCommand(self.handle, body_unique_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REMOVE_BODY_COMPLETED,
        )?;
        Ok(())
    }

    pub fn get_num_constraints(&self) -> i32 {
        unsafe { ffi::b3GetNumUserConstraints(self.handle) }
    }

    pub fn get_constraint_info(&self, constraint_id: i32) -> BulletResult<ConstraintInfo> {
        let mut raw = MaybeUninit::<ffi::b3UserConstraint>::uninit();
        let success =
            unsafe { ffi::b3GetUserConstraintInfo(self.handle, constraint_id, raw.as_mut_ptr()) };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Unable to query constraint info",
                code: success,
            });
        }
        let raw = unsafe { raw.assume_init() };
        Ok(ConstraintInfo {
            parent_body: raw.m_parentBodyIndex,
            parent_link: raw.m_parentJointIndex,
            child_body: raw.m_childBodyIndex,
            child_link: raw.m_childJointIndex,
            parent_frame: Self::read_frame_transform(&raw.m_parentFrame),
            child_frame: Self::read_frame_transform(&raw.m_childFrame),
            joint_axis: raw.m_jointAxis,
            joint_type: raw.m_jointType,
            max_applied_force: raw.m_maxAppliedForce,
            constraint_unique_id: raw.m_userConstraintUniqueId,
            gear_ratio: raw.m_gearRatio,
            gear_aux_link: raw.m_gearAuxLink,
            relative_position_target: raw.m_relativePositionTarget,
            erp: raw.m_erp,
        })
    }

    pub fn get_constraint_state(&mut self, constraint_id: i32) -> BulletResult<ConstraintState> {
        self.ensure_can_submit()?;
        let command =
            unsafe { ffi::b3InitGetUserConstraintStateCommand(self.handle, constraint_id) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_CONSTRAINT_REQUEST_STATE_COMPLETED,
        )?;
        let mut raw = MaybeUninit::<ffi::b3UserConstraintState>::uninit();
        let success =
            unsafe { ffi::b3GetStatusUserConstraintState(status.handle, raw.as_mut_ptr()) };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Unable to read constraint state",
                code: success,
            });
        }
        let raw = unsafe { raw.assume_init() };
        Ok(ConstraintState {
            applied_forces: raw.m_appliedConstraintForces,
            dof_count: raw.m_numDofs,
        })
    }

    pub fn get_constraint_unique_id(&self, index: i32) -> Option<i32> {
        let id = unsafe { ffi::b3GetUserConstraintId(self.handle, index) };
        if id < 0 { None } else { Some(id) }
    }

    pub fn load_texture(&mut self, filename: &str) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let filename_c = CString::new(filename)?;
        let command = unsafe { ffi::b3InitLoadTexture(self.handle, filename_c.as_ptr()) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_LOAD_TEXTURE_COMPLETED,
        )?;
        let texture_id = unsafe { ffi::b3GetStatusTextureUniqueId(status.handle) };
        if texture_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to load texture",
                code: texture_id,
            });
        }
        Ok(texture_id)
    }

    pub fn change_texture(&mut self, texture_id: i32, data: &TextureData<'_>) -> BulletResult<()> {
        if data.width <= 0 || data.height <= 0 {
            return Err(BulletError::CommandFailed {
                message: "Texture dimensions must be positive",
                code: -1,
            });
        }

        let expected_len = (data.width as usize)
            .saturating_mul(data.height as usize)
            .saturating_mul(3);
        if expected_len != data.rgb_pixels.len() {
            return Err(BulletError::CommandFailed {
                message: "Texture data length mismatch",
                code: data.rgb_pixels.len() as i32,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3CreateChangeTextureCommandInit(
                self.handle,
                texture_id,
                data.width,
                data.height,
                data.rgb_pixels.as_ptr().cast(),
            )
        };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }
}

/// ! =====================================================================================================================================
/// ### Bodies, Joints & Base State
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | getNumBodies | **Implemented** | Enumeration helper |
/// | getBodyUniqueId | **Implemented** | Enumeration helper |
/// | getBodyInfo | **Implemented** | Names cached |
/// | computeDofCount | **Implemented** | Useful with dynamics |
/// | syncBodyInfo | **Implemented** | Multi-client support |
/// | syncUserData | **Implemented** | Multi-client support |
/// | addUserData | **Implemented** | User data authoring |
/// | getUserData | **Implemented** | User data query |
/// | removeUserData | **Implemented** | User data cleanup |
/// | getUserDataId | **Implemented** | User data query |
/// | getNumUserData | **Implemented** | User data query |
/// | getUserDataInfo | **Implemented** | User data query |
/// | getBasePositionAndOrientation | **Implemented** | Uses actual state request |
/// | getAABB | **Implemented** | Contact bounds |
/// | resetBasePositionAndOrientation | **Implemented** | World authoring priority |
/// | unsupportedChangeScaling | Optional | Rudimentary scaling |
/// | getBaseVelocity | **Implemented** | Uses actual state request |
/// | resetBaseVelocity | **Implemented** | Dynamics priority |
/// | getNumJoints | **Implemented** | Joint enumeration |
/// | getJointInfo | **Implemented** | Joint metadata |
/// | getJointState | **Implemented** | Single joint sensor |
/// | getJointStates | **Implemented** | Batch sensor support |
/// | getJointStateMultiDof | **Implemented** | Multi-DoF sensor |
/// | getJointStatesMultiDof | **Implemented** | Multi-DoF batch |
/// | getLinkState | **Implemented** | Forward kinematics |
/// | getLinkStates | **Implemented** | Batch link state |
/// | resetJointState | **Implemented** | World authoring priority |
/// | resetJointStateMultiDof | **Implemented** | Multi-DoF reset |
/// | resetJointStatesMultiDof | **Implemented** | Multi-DoF batch reset |
impl PhysicsClient {
    pub fn compute_dof_count(&self, body_unique_id: i32) -> i32 {
        unsafe { ffi::b3ComputeDofCount(self.handle, body_unique_id) }
    }

    pub fn sync_body_info(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitSyncBodyInfoCommand(self.handle) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_SYNC_BODY_INFO_COMPLETED,
        )?;
        Ok(())
    }

    pub fn sync_user_data(&mut self, body_unique_ids: Option<&[i32]>) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitSyncUserDataCommand(self.handle) };
        if let Some(ids) = body_unique_ids {
            for &id in ids {
                unsafe {
                    ffi::b3AddBodyToSyncUserDataRequest(command, id);
                }
            }
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_SYNC_USER_DATA_COMPLETED,
        )?;
        Ok(())
    }

    pub fn add_user_data(
        &mut self,
        body_unique_id: i32,
        key: &str,
        value: &str,
        link_index: Option<i32>,
        visual_shape_index: Option<i32>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let key_c = CString::new(key)?;
        let value_c = CString::new(value)?;
        let link_index = link_index.unwrap_or(-1);
        let visual_shape_index = visual_shape_index.unwrap_or(-1);
        let value_len = i32::try_from(value_c.as_bytes_with_nul().len()).map_err(|_| {
            BulletError::CommandFailed { message: "User data value is too long", code: -1 }
        })?;

        let command = unsafe {
            ffi::b3InitAddUserDataCommand(
                self.handle,
                body_unique_id,
                link_index,
                visual_shape_index,
                key_c.as_ptr(),
                ffi::USER_DATA_VALUE_TYPE_STRING,
                value_len,
                value_c.as_ptr().cast(),
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_ADD_USER_DATA_COMPLETED,
        )?;
        let user_data_id = unsafe { ffi::b3GetUserDataIdFromStatus(status.handle) };
        Ok(user_data_id)
    }

    pub fn remove_user_data(&mut self, user_data_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRemoveUserDataCommand(self.handle, user_data_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REMOVE_USER_DATA_COMPLETED,
        )?;
        Ok(())
    }

    pub fn get_user_data_id(
        &self,
        body_unique_id: i32,
        key: &str,
        link_index: Option<i32>,
        visual_shape_index: Option<i32>,
    ) -> BulletResult<Option<i32>> {
        let key_c = CString::new(key)?;
        let link_index = link_index.unwrap_or(-1);
        let visual_shape_index = visual_shape_index.unwrap_or(-1);
        let id = unsafe {
            ffi::b3GetUserDataId(
                self.handle,
                body_unique_id,
                link_index,
                visual_shape_index,
                key_c.as_ptr(),
            )
        };
        if id < 0 { Ok(None) } else { Ok(Some(id)) }
    }

    pub fn get_user_data(&self, user_data_id: i32) -> BulletResult<Option<String>> {
        let mut raw = MaybeUninit::<ffi::b3UserDataValue>::uninit();
        let success = unsafe { ffi::b3GetUserData(self.handle, user_data_id, raw.as_mut_ptr()) };
        if success == 0 {
            return Ok(None);
        }
        let raw = unsafe { raw.assume_init() };
        if raw.m_type != ffi::USER_DATA_VALUE_TYPE_STRING {
            return Err(BulletError::CommandFailed {
                message: "User data value has unsupported type",
                code: raw.m_type,
            });
        }
        if raw.m_data1.is_null() {
            return Ok(None);
        }
        let value = unsafe { CStr::from_ptr(raw.m_data1) };
        Ok(Some(value.to_string_lossy().into_owned()))
    }

    pub fn get_num_user_data(&self, body_unique_id: i32) -> BulletResult<i32> {
        let count = unsafe { ffi::b3GetNumUserData(self.handle, body_unique_id) };
        if count < 0 {
            Err(BulletError::CommandFailed {
                message: "Failed to query user data count",
                code: count,
            })
        } else {
            Ok(count)
        }
    }

    pub fn get_user_data_info(
        &self,
        body_unique_id: i32,
        user_data_index: i32,
    ) -> BulletResult<UserDataInfo> {
        let mut key_ptr: *const i8 = ptr::null();
        let mut user_data_id = -1;
        let mut link_index = -1;
        let mut visual_shape_index = -1;
        unsafe {
            ffi::b3GetUserDataInfo(
                self.handle,
                body_unique_id,
                user_data_index,
                &raw mut key_ptr,
                &raw mut user_data_id,
                &raw mut link_index,
                &raw mut visual_shape_index,
            );
        }

        if key_ptr.is_null() || user_data_id == -1 {
            return Err(BulletError::CommandFailed {
                message: "Could not fetch user data info",
                code: -1,
            });
        }

        let key = unsafe { CStr::from_ptr(key_ptr) }
            .to_string_lossy()
            .into_owned();

        Ok(UserDataInfo {
            user_data_id,
            key,
            body_unique_id,
            link_index,
            visual_shape_index,
        })
    }

    pub fn get_aabb(&mut self, body_unique_id: i32, link_index: i32) -> BulletResult<Aabb> {
        if body_unique_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Invalid body id",
                code: body_unique_id,
            });
        }
        if link_index < -1 {
            return Err(BulletError::CommandFailed {
                message: "Invalid link index",
                code: link_index,
            });
        }
        self.ensure_can_submit()?;
        let command =
            unsafe { ffi::b3RequestCollisionInfoCommandInit(self.handle, body_unique_id) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_COLLISION_INFO_COMPLETED,
        )?;

        let mut aabb_min = [0.0; 3];
        let mut aabb_max = [0.0; 3];
        let success = unsafe {
            ffi::b3GetStatusAABB(
                status.handle,
                link_index,
                aabb_min.as_mut_ptr(),
                aabb_max.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to fetch AABB",
                code: success,
            });
        }

        Ok(Aabb { min: aabb_min, max: aabb_max })
    }

    pub fn reset_base_position_and_orientation(
        &mut self,
        body_unique_id: i32,
        position: [f64; 3],
        orientation: [f64; 4],
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreatePoseCommandInit(self.handle, body_unique_id) };
        unsafe {
            ffi::b3CreatePoseCommandSetBasePosition(command, position[0], position[1], position[2]);
            ffi::b3CreatePoseCommandSetBaseOrientation(
                command,
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3],
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn reset_base_velocity(
        &mut self,
        body_unique_id: i32,
        linear_velocity: Option<[f64; 3]>,
        angular_velocity: Option<[f64; 3]>,
    ) -> BulletResult<()> {
        if linear_velocity.is_none() && angular_velocity.is_none() {
            return Err(BulletError::CommandFailed {
                message: "Expected linear and/or angular velocity",
                code: -1,
            });
        }
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreatePoseCommandInit(self.handle, body_unique_id) };
        if let Some(vel) = linear_velocity {
            unsafe {
                ffi::b3CreatePoseCommandSetBaseLinearVelocity(command, vel.as_ptr());
            }
        }
        if let Some(vel) = angular_velocity {
            unsafe {
                ffi::b3CreatePoseCommandSetBaseAngularVelocity(command, vel.as_ptr());
            }
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn get_num_bodies(&self) -> i32 {
        unsafe { ffi::b3GetNumBodies(self.handle) }
    }

    pub fn get_body_unique_id(&self, serial_index: i32) -> i32 {
        unsafe { ffi::b3GetBodyUniqueId(self.handle, serial_index) }
    }

    pub fn get_body_info(&self, body_unique_id: i32) -> BulletResult<BodyInfo> {
        let mut raw = MaybeUninit::<ffi::b3BodyInfo>::uninit();
        let result = unsafe { ffi::b3GetBodyInfo(self.handle, body_unique_id, raw.as_mut_ptr()) };
        if result == 0 {
            return Err(BulletError::CommandFailed {
                message: "Cannot query body info",
                code: result,
            });
        }
        let raw = unsafe { raw.assume_init() };
        let base_name = Self::read_c_string(&raw.m_baseName);
        let body_name = Self::read_c_string(&raw.m_bodyName);
        Ok(BodyInfo { base_name, body_name })
    }

    pub fn get_num_joints(&self, body_unique_id: i32) -> i32 {
        unsafe { ffi::b3GetNumJoints(self.handle, body_unique_id) }
    }

    pub fn get_joint_info(&self, body_unique_id: i32, joint_index: i32) -> BulletResult<JointInfo> {
        let mut raw = MaybeUninit::<ffi::b3JointInfo>::uninit();
        let success = unsafe {
            ffi::b3GetJointInfo(self.handle, body_unique_id, joint_index, raw.as_mut_ptr())
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Cannot query joint info",
                code: success,
            });
        }
        unsafe { raw.assume_init() }.try_into()
    }

    pub fn get_joint_state(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
    ) -> BulletResult<JointState> {
        let status = self.request_actual_state_status(body_unique_id)?;
        self.read_joint_state(status.handle, joint_index)
    }

    pub fn get_joint_states(
        &mut self,
        body_unique_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<Vec<JointState>> {
        if joint_indices.is_empty() {
            return Ok(Vec::new());
        }
        let status = self.request_actual_state_status(body_unique_id)?;
        joint_indices
            .iter()
            .map(|&index| self.read_joint_state(status.handle, index))
            .collect()
    }

    pub fn get_joint_state_multi_dof(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
    ) -> BulletResult<JointStateMultiDof> {
        let status = self.request_actual_state_status(body_unique_id)?;
        self.read_joint_state_multi_dof(status.handle, joint_index)
    }

    pub fn get_joint_states_multi_dof(
        &mut self,
        body_unique_id: i32,
        joint_indices: &[i32],
    ) -> BulletResult<Vec<JointStateMultiDof>> {
        if joint_indices.is_empty() {
            return Ok(Vec::new());
        }
        let status = self.request_actual_state_status(body_unique_id)?;
        joint_indices
            .iter()
            .map(|&index| self.read_joint_state_multi_dof(status.handle, index))
            .collect()
    }

    pub fn get_base_position_and_orientation(
        &mut self,
        body_unique_id: i32,
    ) -> BulletResult<na::Isometry3<f64>> {
        let status = self.request_actual_state_status(body_unique_id)?;
        let (base, _) = Self::extract_base_state(status.handle)?;
        Ok(base)
    }

    pub fn get_base_velocity(&mut self, body_unique_id: i32) -> BulletResult<[f64; 6]> {
        let status = self.request_actual_state_status(body_unique_id)?;
        let (_, velocity) = Self::extract_base_state(status.handle)?;
        Ok(velocity)
    }

    pub fn get_link_state(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        compute_forward_kinematics: bool,
        compute_link_velocity: bool,
    ) -> BulletResult<LinkState> {
        let status = self.request_actual_state_status_with_flags(
            body_unique_id,
            compute_link_velocity,
            compute_forward_kinematics,
        )?;
        self.read_link_state(status.handle, link_index)
    }

    pub fn get_link_states(
        &mut self,
        body_unique_id: i32,
        link_indices: &[i32],
        compute_forward_kinematics: bool,
        compute_link_velocity: bool,
    ) -> BulletResult<Vec<LinkState>> {
        if link_indices.is_empty() {
            return Ok(Vec::new());
        }
        let status = self.request_actual_state_status_with_flags(
            body_unique_id,
            compute_link_velocity,
            compute_forward_kinematics,
        )?;
        link_indices
            .iter()
            .map(|&index| self.read_link_state(status.handle, index))
            .collect()
    }

    pub fn reset_joint_state(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
        target_position: f64,
        target_velocity: Option<f64>,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreatePoseCommandInit(self.handle, body_unique_id) };
        unsafe {
            ffi::b3CreatePoseCommandSetJointPosition(
                self.handle,
                command,
                joint_index,
                target_position,
            );
            ffi::b3CreatePoseCommandSetJointVelocity(
                self.handle,
                command,
                joint_index,
                target_velocity.unwrap_or(0.0),
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn reset_joint_state_multi_dof(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
        positions: Option<&[f64]>,
        velocities: Option<&[f64]>,
    ) -> BulletResult<()> {
        let target = MultiDofTarget { joint_index, positions, velocities };
        self.reset_joint_states_multi_dof(body_unique_id, slice::from_ref(&target))
    }

    pub fn reset_joint_states_multi_dof(
        &mut self,
        body_unique_id: i32,
        targets: &[MultiDofTarget<'_>],
    ) -> BulletResult<()> {
        if targets.is_empty() {
            return Ok(());
        }
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreatePoseCommandInit(self.handle, body_unique_id) };

        for target in targets {
            let mut position_buf = [0.0_f64; 4];
            let mut velocity_buf = [0.0_f64; 3];
            let mut position_size = 0_i32;
            let mut velocity_size = 0_i32;

            if let Some(positions) = target.positions {
                if positions.len() > position_buf.len() {
                    return Err(BulletError::CommandFailed {
                        message: "Position vector too large for multi-DoF joint",
                        code: positions.len() as i32,
                    });
                }
                if !positions.is_empty() {
                    position_buf[..positions.len()].copy_from_slice(positions);
                    position_size = positions.len() as i32;
                }
            }

            if let Some(velocities) = target.velocities {
                if velocities.len() > velocity_buf.len() {
                    return Err(BulletError::CommandFailed {
                        message: "Velocity vector too large for multi-DoF joint",
                        code: velocities.len() as i32,
                    });
                }
                if !velocities.is_empty() {
                    velocity_buf[..velocities.len()].copy_from_slice(velocities);
                    velocity_size = velocities.len() as i32;
                }
            }

            if position_size == 0 && velocity_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Expected position and/or velocity data for multi-DoF joint",
                    code: target.joint_index,
                });
            }

            if position_size > 0 {
                unsafe {
                    ffi::b3CreatePoseCommandSetJointPositionMultiDof(
                        self.handle,
                        command,
                        target.joint_index,
                        position_buf.as_ptr(),
                        position_size,
                    );
                }
            }

            if velocity_size > 0 {
                unsafe {
                    ffi::b3CreatePoseCommandSetJointVelocityMultiDof(
                        self.handle,
                        command,
                        target.joint_index,
                        velocity_buf.as_ptr(),
                        velocity_size,
                    );
                }
            }
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }
}

/// ! =====================================================================================================================================
/// ### Dynamics & Control
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | changeDynamics | **Implemented** | Mutation wrapper |
/// | getDynamicsInfo | **Implemented** | Mirrors Bullet query |
/// | setJointMotorControl | Optional | Legacy single-call (deprecated) |
/// | setJointMotorControl2 | **Implemented** | Primary motor control path |
/// | setJointMotorControlMultiDof | **Implemented** | Multi-DoF control |
/// | setJointMotorControlMultiDofArray | **Implemented** | Multi-DoF batch control |
/// | setJointMotorControlArray | **Implemented** | Batch joint control |
/// | applyExternalForce | **Implemented** | Core dynamics action |
/// | applyExternalTorque | **Implemented** | Core dynamics action |
/// | calculateInverseDynamics | **Implemented** | Advanced dynamics |
/// | calculateJacobian | **Implemented** | Advanced dynamics |
/// | calculateMassMatrix | **Implemented** | Advanced dynamics |
/// | calculateInverseKinematics | **Implemented** | Depends on jacobians |
/// | calculateInverseKinematics2 | **Implemented** | Multi-end-effector variant |
impl PhysicsClient {
    pub fn change_dynamics(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        update: &DynamicsUpdate,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitChangeDynamicsInfo(self.handle) };

        unsafe {
            if let Some(mass) = update.mass {
                ffi::b3ChangeDynamicsInfoSetMass(command, body_unique_id, link_index, mass);
            }
            if let Some(local_inertia) = update.local_inertia_diagonal {
                ffi::b3ChangeDynamicsInfoSetLocalInertiaDiagonal(
                    command,
                    body_unique_id,
                    link_index,
                    local_inertia.as_ptr(),
                );
            }
            if let Some(anisotropic) = update.anisotropic_friction {
                ffi::b3ChangeDynamicsInfoSetAnisotropicFriction(
                    command,
                    body_unique_id,
                    link_index,
                    anisotropic.as_ptr(),
                );
            }
            if let Some((lower, upper)) = update.joint_limit {
                ffi::b3ChangeDynamicsInfoSetJointLimit(
                    command,
                    body_unique_id,
                    link_index,
                    lower,
                    upper,
                );
            }
            if let Some(limit_force) = update.joint_limit_force {
                ffi::b3ChangeDynamicsInfoSetJointLimitForce(
                    command,
                    body_unique_id,
                    link_index,
                    limit_force,
                );
            }
            if let Some(dynamic_type) = update.dynamic_type {
                ffi::b3ChangeDynamicsInfoSetDynamicType(
                    command,
                    body_unique_id,
                    link_index,
                    dynamic_type,
                );
            }
            if let Some(lateral) = update.lateral_friction {
                ffi::b3ChangeDynamicsInfoSetLateralFriction(
                    command,
                    body_unique_id,
                    link_index,
                    lateral,
                );
            }
            if let Some(spinning) = update.spinning_friction {
                ffi::b3ChangeDynamicsInfoSetSpinningFriction(
                    command,
                    body_unique_id,
                    link_index,
                    spinning,
                );
            }
            if let Some(rolling) = update.rolling_friction {
                ffi::b3ChangeDynamicsInfoSetRollingFriction(
                    command,
                    body_unique_id,
                    link_index,
                    rolling,
                );
            }
            if let Some(restitution) = update.restitution {
                ffi::b3ChangeDynamicsInfoSetRestitution(
                    command,
                    body_unique_id,
                    link_index,
                    restitution,
                );
            }
            if let Some(linear_damping) = update.linear_damping {
                ffi::b3ChangeDynamicsInfoSetLinearDamping(command, body_unique_id, linear_damping);
            }
            if let Some(angular_damping) = update.angular_damping {
                ffi::b3ChangeDynamicsInfoSetAngularDamping(
                    command,
                    body_unique_id,
                    angular_damping,
                );
            }
            if let Some(joint_damping) = update.joint_damping {
                ffi::b3ChangeDynamicsInfoSetJointDamping(
                    command,
                    body_unique_id,
                    link_index,
                    joint_damping,
                );
            }
            if let Some((stiffness, damping)) = update.contact_stiffness_and_damping {
                ffi::b3ChangeDynamicsInfoSetContactStiffnessAndDamping(
                    command,
                    body_unique_id,
                    link_index,
                    stiffness,
                    damping,
                );
            }
            if let Some(anchor) = update.friction_anchor {
                ffi::b3ChangeDynamicsInfoSetFrictionAnchor(
                    command,
                    body_unique_id,
                    link_index,
                    anchor as i32,
                );
            }
            if let Some(radius) = update.ccd_swept_sphere_radius {
                ffi::b3ChangeDynamicsInfoSetCcdSweptSphereRadius(
                    command,
                    body_unique_id,
                    link_index,
                    radius,
                );
            }
            if let Some(threshold) = update.contact_processing_threshold {
                ffi::b3ChangeDynamicsInfoSetContactProcessingThreshold(
                    command,
                    body_unique_id,
                    link_index,
                    threshold,
                );
            }
            if let Some(state) = update.activation_state {
                ffi::b3ChangeDynamicsInfoSetActivationState(command, body_unique_id, state);
            }
            if let Some(max_velocity) = update.max_joint_velocity {
                ffi::b3ChangeDynamicsInfoSetMaxJointVelocity(command, body_unique_id, max_velocity);
            }
            if let Some(collision_margin) = update.collision_margin {
                ffi::b3ChangeDynamicsInfoSetCollisionMargin(
                    command,
                    body_unique_id,
                    collision_margin,
                );
            }
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn get_dynamics_info(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
    ) -> BulletResult<DynamicsInfo> {
        self.ensure_can_submit()?;
        let command =
            unsafe { ffi::b3GetDynamicsInfoCommandInit(self.handle, body_unique_id, link_index) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_GET_DYNAMICS_INFO_COMPLETED,
        )?;

        let mut info = MaybeUninit::<ffi::b3DynamicsInfo>::uninit();
        let success = unsafe { ffi::b3GetDynamicsInfo(status.handle, info.as_mut_ptr()) };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to retrieve dynamics info",
                code: success,
            });
        }
        let info = unsafe { info.assume_init() };
        Ok(DynamicsInfo {
            mass: info.m_mass,
            lateral_friction: info.m_lateralFrictionCoeff,
            local_inertia_diagonal: info.m_localInertialDiagonal,
            local_inertia_position: [
                info.m_localInertialFrame[0],
                info.m_localInertialFrame[1],
                info.m_localInertialFrame[2],
            ],
            local_inertia_orientation: [
                info.m_localInertialFrame[3],
                info.m_localInertialFrame[4],
                info.m_localInertialFrame[5],
                info.m_localInertialFrame[6],
            ],
            restitution: info.m_restitution,
            rolling_friction: info.m_rollingFrictionCoeff,
            spinning_friction: info.m_spinningFrictionCoeff,
            contact_damping: info.m_contactDamping,
            contact_stiffness: info.m_contactStiffness,
            body_type: info.m_bodyType.try_into()?,
            collision_margin: info.m_collisionMargin,
            angular_damping: info.m_angularDamping,
            linear_damping: info.m_linearDamping,
            ccd_swept_sphere_radius: info.m_ccdSweptSphereRadius,
            contact_processing_threshold: info.m_contactProcessingThreshold,
            activation_state: info.m_activationState,
            friction_anchor: info.m_frictionAnchor != 0,
            dynamic_type: info.m_dynamicType,
        })
    }

    pub fn set_joint_motor_control(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
        control_mode: ControlMode,
        maximum_force: Option<f64>,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let info = self.get_joint_info(body_unique_id, joint_index)?;
        let command = unsafe {
            ffi::b3JointControlCommandInit2(self.handle, body_unique_id, control_mode.as_raw())
        };

        let u_index = info.u_index;
        if u_index < 0 {
            return Err(BulletError::CommandFailed {
                message: "Joint has no velocity DOF for motor control",
                code: u_index,
            });
        }

        let force = maximum_force.unwrap_or(100_000.);
        let kp = 0.1;
        let kd = 1.0;
        let target_velocity = 0.;

        unsafe {
            match control_mode {
                ControlMode::Position(target_position) => {
                    ffi::b3JointControlSetDesiredPosition(command, info.q_index, target_position);

                    ffi::b3JointControlSetKp(command, info.u_index, kp);
                    ffi::b3JointControlSetDesiredVelocity(command, info.u_index, target_velocity);

                    ffi::b3JointControlSetKd(command, info.u_index, kd);
                    ffi::b3JointControlSetMaximumForce(command, info.u_index, force);
                }
                ControlMode::Pd {
                    target_position: pos,
                    target_velocity: vel,
                    position_gain: kp,
                    velocity_gain: kd,
                    maximum_velocity: max_vel,
                }
                | ControlMode::StablePd {
                    target_position: pos,
                    target_velocity: vel,
                    position_gain: kp,
                    velocity_gain: kd,
                    maximum_velocity: max_vel,
                } => {
                    if let Some(max_vel) = max_vel {
                        ffi::b3JointControlSetMaximumVelocity(command, info.u_index, max_vel);
                    }
                    ffi::b3JointControlSetDesiredPosition(command, info.q_index, pos);

                    ffi::b3JointControlSetKp(command, info.u_index, kp);
                    ffi::b3JointControlSetDesiredVelocity(command, info.u_index, vel);

                    ffi::b3JointControlSetKd(command, info.u_index, kd);
                    ffi::b3JointControlSetMaximumForce(command, info.u_index, force);
                }
                ControlMode::Velocity(vel) => {
                    ffi::b3JointControlSetDesiredVelocity(command, info.u_index, vel);
                    ffi::b3JointControlSetKd(command, info.u_index, kd);
                    ffi::b3JointControlSetMaximumForce(command, info.u_index, force);
                }
                ControlMode::Torque(f) => {
                    ffi::b3JointControlSetDesiredForceTorque(command, info.u_index, f);
                }
            }
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn set_joint_motor_control_array(
        &mut self,
        body_unique_id: i32,
        joint_index: &[i32],
        control_mode: ControlModeArray<'_>,
        maximum_force: Option<&[f64]>,
    ) -> BulletResult<()> {
        const DEFAULT_FORCE: f64 = 100_000.0;
        const DEFAULT_KP: f64 = 0.1;
        const DEFAULT_KD: f64 = 1.0;
        const DEFAULT_TARGET_VELOCITY: f64 = 0.0;

        if joint_index.is_empty() {
            return Ok(());
        }
        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3JointControlCommandInit2(self.handle, body_unique_id, control_mode.as_raw())
        };

        let joint_count = joint_index.len();

        match control_mode {
            ControlModeArray::Position(targets) => {
                if targets.len() != joint_count {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and position target lengths differ",
                        code: -1,
                    });
                }
                if let Some(forces) = maximum_force
                    && forces.len() != joint_count
                {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and maximum force lengths differ",
                        code: -1,
                    });
                }

                for (i, joint) in joint_index.iter().copied().enumerate() {
                    let info = self.get_joint_info(body_unique_id, joint)?;
                    unsafe {
                        ffi::b3JointControlSetDesiredPosition(command, info.q_index, targets[i]);
                        ffi::b3JointControlSetKp(command, info.u_index, DEFAULT_KP);
                        ffi::b3JointControlSetDesiredVelocity(
                            command,
                            info.u_index,
                            DEFAULT_TARGET_VELOCITY,
                        );
                        ffi::b3JointControlSetKd(command, info.u_index, DEFAULT_KD);
                        ffi::b3JointControlSetMaximumForce(
                            command,
                            info.u_index,
                            maximum_force.map_or(DEFAULT_FORCE, |forces| forces[i]),
                        );
                    }
                }
            }
            ControlModeArray::Pd {
                target_position: pos,
                target_velocity: vel,
                position_gain: kp,
                velocity_gain: kd,
                maximum_velocity: max_vel,
            }
            | ControlModeArray::StablePd {
                target_position: pos,
                target_velocity: vel,
                position_gain: kp,
                velocity_gain: kd,
                maximum_velocity: max_vel,
            } => {
                if pos.len() != joint_count
                    || vel.len() != joint_count
                    || kp.len() != joint_count
                    || kd.len() != joint_count
                {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and PD target lengths differ",
                        code: -1,
                    });
                }
                if let Some(max_velocities) = max_vel
                    && max_velocities.len() != joint_count
                {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and maximum velocity lengths differ",
                        code: -1,
                    });
                }

                if let Some(forces) = maximum_force
                    && forces.len() != joint_count
                {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and maximum force lengths differ",
                        code: -1,
                    });
                }

                for (i, joint) in joint_index.iter().copied().enumerate() {
                    let info = self.get_joint_info(body_unique_id, joint)?;
                    if let Some(max_velocities) = max_vel {
                        unsafe {
                            ffi::b3JointControlSetMaximumVelocity(
                                command,
                                info.u_index,
                                max_velocities[i],
                            );
                        }
                    }
                    unsafe {
                        ffi::b3JointControlSetDesiredPosition(command, info.q_index, pos[i]);
                        ffi::b3JointControlSetKp(command, info.u_index, kp[i]);
                        ffi::b3JointControlSetDesiredVelocity(command, info.u_index, vel[i]);
                        ffi::b3JointControlSetKd(command, info.u_index, kd[i]);
                        ffi::b3JointControlSetMaximumForce(
                            command,
                            info.u_index,
                            maximum_force.map_or(DEFAULT_FORCE, |forces| forces[i]),
                        );
                    }
                }
            }
            ControlModeArray::Velocity(velocities) => {
                if velocities.len() != joint_count {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and velocity target lengths differ",
                        code: -1,
                    });
                }
                if let Some(forces) = maximum_force
                    && forces.len() != joint_count
                {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and maximum force lengths differ",
                        code: -1,
                    });
                }

                for (i, joint) in joint_index.iter().copied().enumerate() {
                    let info = self.get_joint_info(body_unique_id, joint)?;
                    unsafe {
                        ffi::b3JointControlSetDesiredVelocity(command, info.u_index, velocities[i]);
                        ffi::b3JointControlSetKd(command, info.u_index, DEFAULT_KD);
                        ffi::b3JointControlSetMaximumForce(
                            command,
                            info.u_index,
                            maximum_force.map_or(DEFAULT_FORCE, |forces| forces[i]),
                        );
                    }
                }
            }
            ControlModeArray::Torque(forces) => {
                if forces.len() != joint_count {
                    return Err(BulletError::CommandFailed {
                        message: "Joint index and torque lengths differ",
                        code: -1,
                    });
                }

                for (force, joint) in forces.iter().copied().zip(joint_index.iter().copied()) {
                    let info = self.get_joint_info(body_unique_id, joint)?;
                    unsafe {
                        ffi::b3JointControlSetDesiredForceTorque(command, info.u_index, force);
                    }
                }
            }
        }

        self.submit_command(command)?;
        Ok(())
    }

    pub fn set_joint_motor_control_multi_dof(
        &mut self,
        body_unique_id: i32,
        joint_index: i32,
        control: &MultiDofControl<'_>,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let info = self.get_joint_info(body_unique_id, joint_index)?;
        let command = unsafe {
            ffi::b3JointControlCommandInit2(self.handle, body_unique_id, control.as_raw())
        };

        Self::configure_multi_dof_control(command, &info, control)?;

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn set_joint_motor_control_multi_dof_array(
        &mut self,
        body_unique_id: i32,
        entries: &[MultiDofControlEntry<'_>],
    ) -> BulletResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.ensure_can_submit()?;
        // 选择第一个条目的控制模式初始化命令（Bullet 要求一次命令同一模式）。
        let command = unsafe {
            ffi::b3JointControlCommandInit2(
                self.handle,
                body_unique_id,
                entries[0].control.as_raw(),
            )
        };

        // 校验所有条目控制模式一致，以符合底层一次命令的限制。
        let expected_mode = entries[0].control.as_raw();
        for entry in entries {
            if entry.control.as_raw() != expected_mode {
                return Err(BulletError::CommandFailed {
                    message: "All Multi-DoF array entries must use the same control mode",
                    code: -1,
                });
            }
            let info = self.get_joint_info(body_unique_id, entry.joint_index)?;
            Self::configure_multi_dof_control(command, &info, &entry.control)?;
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn apply_external_force(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        force: impl Into<[f64; 3]>,
        position: impl Into<[f64; 3]>,
        frame: ForceFrame,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3ApplyExternalForceCommandInit(self.handle) };
        let force = force.into();
        let position = position.into();
        unsafe {
            ffi::b3ApplyExternalForce(
                command,
                body_unique_id,
                link_index,
                force.as_ptr(),
                position.as_ptr(),
                frame.as_raw(),
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn apply_external_torque(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        torque: impl Into<[f64; 3]>,
        frame: ForceFrame,
    ) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3ApplyExternalForceCommandInit(self.handle) };
        let torque = torque.into();
        unsafe {
            ffi::b3ApplyExternalTorque(
                command,
                body_unique_id,
                link_index,
                torque.as_ptr(),
                frame.as_raw(),
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn calculate_inverse_dynamics(
        &mut self,
        body_unique_id: i32,
        joint_positions: &[f64],
        joint_velocities: &[f64],
        joint_accelerations: &[f64],
    ) -> BulletResult<Vec<f64>> {
        if joint_positions.len() != joint_velocities.len()
            || joint_positions.len() != joint_accelerations.len()
        {
            return Err(BulletError::CommandFailed {
                message: "Inverse dynamics expects position/velocity/acceleration arrays of equal length",
                code: joint_positions.len() as i32,
            });
        }

        self.ensure_can_submit()?;
        let dof = joint_positions.len() as i32;
        let command = unsafe {
            ffi::b3CalculateInverseDynamicsCommandInit2(
                self.handle,
                body_unique_id,
                joint_positions.as_ptr(),
                dof,
                joint_velocities.as_ptr(),
                joint_accelerations.as_ptr(),
                dof,
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CALCULATED_INVERSE_DYNAMICS_COMPLETED,
        )?;

        let mut result_body = 0;
        let mut result_dof = 0;
        let mut joint_forces = vec![0.0_f64; joint_positions.len()];
        let success = unsafe {
            ffi::b3GetStatusInverseDynamicsJointForces(
                status.handle,
                &raw mut result_body,
                &raw mut result_dof,
                joint_forces.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to calculate inverse dynamics",
                code: success,
            });
        }
        joint_forces.truncate(result_dof as usize);
        Ok(joint_forces)
    }

    pub fn calculate_jacobian(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        local_position: [f64; 3],
        joint_positions: &[f64],
        joint_velocities: &[f64],
        joint_accelerations: &[f64],
    ) -> BulletResult<Jacobian> {
        let dof = joint_positions.len();
        if dof == 0 || joint_velocities.len() != dof || joint_accelerations.len() != dof {
            return Err(BulletError::CommandFailed {
                message: "Jacobian computation expects non-empty arrays of equal length",
                code: dof as i32,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3CalculateJacobianCommandInit(
                self.handle,
                body_unique_id,
                link_index,
                local_position.as_ptr(),
                joint_positions.as_ptr(),
                joint_velocities.as_ptr(),
                joint_accelerations.as_ptr(),
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CALCULATED_JACOBIAN_COMPLETED,
        )?;

        let mut out_dof = 0;
        let mut linear = vec![0.0_f64; 3 * dof];
        let mut angular = vec![0.0_f64; 3 * dof];
        let success = unsafe {
            ffi::b3GetStatusJacobian(
                status.handle,
                &raw mut out_dof,
                linear.as_mut_ptr(),
                angular.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to calculate Jacobian",
                code: success,
            });
        }
        let out_cols = out_dof as usize;
        if out_cols == 0 {
            return Ok(Jacobian::zeros(0));
        }

        let mut linear_matrix = na::Matrix3xX::zeros(out_cols);
        let mut angular_matrix = na::Matrix3xX::zeros(out_cols);
        for col in 0..out_cols {
            for row in 0..3 {
                linear_matrix[(row, col)] = linear[row * out_cols + col];
                angular_matrix[(row, col)] = angular[row * out_cols + col];
            }
        }

        Ok(Jacobian { linear: linear_matrix, angular: angular_matrix })
    }

    pub fn calculate_mass_matrix(
        &mut self,
        body_unique_id: i32,
        joint_positions: &[f64],
    ) -> BulletResult<MassMatrix> {
        let dof = joint_positions.len();
        if dof == 0 {
            return Err(BulletError::CommandFailed {
                message: "Mass matrix requires at least one joint coordinate",
                code: dof as i32,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3CalculateMassMatrixCommandInit(
                self.handle,
                body_unique_id,
                joint_positions.as_ptr(),
                dof as i32,
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CALCULATED_MASS_MATRIX_COMPLETED,
        )?;

        let mut out_dof = 0;
        let mut raw = vec![0.0_f64; dof * dof];
        let success = unsafe {
            ffi::b3GetStatusMassMatrix(
                self.handle,
                status.handle,
                &raw mut out_dof,
                raw.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to calculate mass matrix",
                code: success,
            });
        }
        let matrix_size = out_dof as usize;
        if matrix_size == 0 {
            return Ok(MassMatrix { matrix: na::DMatrix::zeros(0, 0) });
        }
        raw.truncate(matrix_size * matrix_size);
        let matrix = na::DMatrix::from_column_slice(matrix_size, matrix_size, &raw);
        Ok(MassMatrix { matrix })
    }

    pub fn calculate_inverse_kinematics(
        &mut self,
        body_unique_id: i32,
        end_effector_link_index: i32,
        target_position: impl Into<na::Isometry3<f64>>,
        options: &InverseKinematicsOptions<'_>,
    ) -> BulletResult<Vec<f64>> {
        self.ensure_can_submit()?;
        let dof = unsafe { ffi::b3ComputeDofCount(self.handle, body_unique_id) };
        if dof <= 0 {
            return Err(BulletError::CommandFailed {
                message: "Body has no movable joints for inverse kinematics",
                code: dof,
            });
        }
        let dof_usize = dof as usize;
        let (tran, rot) = isometry_to_raw_parts(&target_position.into());

        let command =
            unsafe { ffi::b3CalculateInverseKinematicsCommandInit(self.handle, body_unique_id) };

        unsafe { ffi::b3CalculateInverseKinematicsSelectSolver(command, options.solver as i32) };

        if let Some(current_positions) = options.current_positions {
            if current_positions.len() != dof_usize {
                return Err(BulletError::CommandFailed {
                    message: "current_positions length must match degrees of freedom",
                    code: current_positions.len() as i32,
                });
            }
            unsafe {
                ffi::b3CalculateInverseKinematicsSetCurrentPositions(
                    command,
                    dof,
                    current_positions.as_ptr(),
                );
            }
        }
        if let Some(max_iterations) = options.max_iterations
            && max_iterations > 0
        {
            unsafe {
                ffi::b3CalculateInverseKinematicsSetMaxNumIterations(command, max_iterations);
            }
        }
        if let Some(residual_threshold) = options.residual_threshold
            && residual_threshold >= 0.0
        {
            unsafe {
                ffi::b3CalculateInverseKinematicsSetResidualThreshold(command, residual_threshold);
            }
        }

        let has_nullspace = options.lower_limits.map_or(0, <[f64]>::len) == dof_usize
            && options.upper_limits.map_or(0, <[f64]>::len) == dof_usize
            && options.joint_ranges.map_or(0, <[f64]>::len) == dof_usize
            && options.rest_poses.map_or(0, <[f64]>::len) == dof_usize;

        unsafe {
            if has_nullspace {
                ffi::b3CalculateInverseKinematicsPosOrnWithNullSpaceVel(
                    command,
                    dof,
                    end_effector_link_index,
                    tran.as_ptr(),
                    rot.as_ptr(),
                    options.lower_limits.unwrap().as_ptr(),
                    options.upper_limits.unwrap().as_ptr(),
                    options.joint_ranges.unwrap().as_ptr(),
                    options.rest_poses.unwrap().as_ptr(),
                );
            } else {
                ffi::b3CalculateInverseKinematicsAddTargetPositionWithOrientation(
                    command,
                    end_effector_link_index,
                    tran.as_ptr(),
                    rot.as_ptr(),
                );
            }
        }

        if let Some(joint_damping) = options.joint_damping {
            if joint_damping.len() < dof_usize {
                return Err(BulletError::CommandFailed {
                    message: "joint_damping length must be >= degrees of freedom",
                    code: joint_damping.len() as i32,
                });
            }
            unsafe {
                ffi::b3CalculateInverseKinematicsSetJointDamping(
                    command,
                    dof,
                    joint_damping.as_ptr(),
                );
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CALCULATE_INVERSE_KINEMATICS_COMPLETED,
        )?;

        let mut result_body = 0;
        let mut result_dof = 0;
        let mut results = vec![0.0_f64; dof_usize];
        let success = unsafe {
            ffi::b3GetStatusInverseKinematicsJointPositions(
                status.handle,
                &raw mut result_body,
                &raw mut result_dof,
                results.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to compute inverse kinematics",
                code: success,
            });
        }
        results.truncate(result_dof as usize);
        Ok(results)
    }

    pub fn calculate_inverse_kinematics2(
        &mut self,
        body_unique_id: i32,
        options: &InverseKinematicsMultiTargetOptions<'_>,
    ) -> BulletResult<Vec<f64>> {
        self.ensure_can_submit()?;
        let num_targets = options.end_effector_link_indices.len();
        if num_targets == 0 {
            return Err(BulletError::CommandFailed {
                message: "inverse_kinematics2 requires at least one end-effector target",
                code: 0,
            });
        }
        if num_targets != options.target_positions.len() {
            return Err(BulletError::CommandFailed {
                message: "endEffectorLinkIndices and targetPositions must have matching lengths",
                code: num_targets as i32,
            });
        }

        let dof = unsafe { ffi::b3ComputeDofCount(self.handle, body_unique_id) };
        if dof <= 0 {
            return Err(BulletError::CommandFailed {
                message: "Body has no movable joints for inverse kinematics",
                code: dof,
            });
        }
        let dof_usize = dof as usize;

        let command =
            unsafe { ffi::b3CalculateInverseKinematicsCommandInit(self.handle, body_unique_id) };
        if let Some(solver) = options.solver {
            unsafe { ffi::b3CalculateInverseKinematicsSelectSolver(command, solver) };
        }
        if let Some(current_positions) = options.current_positions {
            if current_positions.len() != dof_usize {
                return Err(BulletError::CommandFailed {
                    message: "current_positions length must match degrees of freedom",
                    code: current_positions.len() as i32,
                });
            }
            unsafe {
                ffi::b3CalculateInverseKinematicsSetCurrentPositions(
                    command,
                    dof,
                    current_positions.as_ptr(),
                );
            }
        }
        if let Some(max_iterations) = options.max_iterations
            && max_iterations > 0
        {
            unsafe {
                ffi::b3CalculateInverseKinematicsSetMaxNumIterations(command, max_iterations);
            }
        }
        if let Some(residual_threshold) = options.residual_threshold
            && residual_threshold >= 0.0
        {
            unsafe {
                ffi::b3CalculateInverseKinematicsSetResidualThreshold(command, residual_threshold);
            }
        }

        let mut flattened_positions = Vec::with_capacity(num_targets * 3);
        for pos in options.target_positions {
            flattened_positions.extend_from_slice(pos);
        }
        let link_indices: Vec<i32> = options.end_effector_link_indices.to_vec();

        unsafe {
            ffi::b3CalculateInverseKinematicsAddTargetsPurePosition(
                command,
                link_indices.len() as i32,
                link_indices.as_ptr(),
                flattened_positions.as_ptr(),
            );
        }

        let has_nullspace = options.lower_limits.map_or(0, <[f64]>::len) == dof_usize
            && options.upper_limits.map_or(0, <[f64]>::len) == dof_usize
            && options.joint_ranges.map_or(0, <[f64]>::len) == dof_usize
            && options.rest_poses.map_or(0, <[f64]>::len) == dof_usize;

        unsafe {
            if has_nullspace {
                ffi::b3CalculateInverseKinematicsPosWithNullSpaceVel(
                    command,
                    dof,
                    link_indices[0],
                    options.target_positions[0].as_ptr(),
                    options.lower_limits.unwrap().as_ptr(),
                    options.upper_limits.unwrap().as_ptr(),
                    options.joint_ranges.unwrap().as_ptr(),
                    options.rest_poses.unwrap().as_ptr(),
                );
            }
        }

        if let Some(joint_damping) = options.joint_damping {
            if joint_damping.len() < dof_usize {
                return Err(BulletError::CommandFailed {
                    message: "joint_damping length must be >= degrees of freedom",
                    code: joint_damping.len() as i32,
                });
            }
            unsafe {
                ffi::b3CalculateInverseKinematicsSetJointDamping(
                    command,
                    dof,
                    joint_damping.as_ptr(),
                );
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CALCULATE_INVERSE_KINEMATICS_COMPLETED,
        )?;

        let mut result_body = 0;
        let mut result_dof = 0;
        let mut results = vec![0.0_f64; dof_usize];
        let success = unsafe {
            ffi::b3GetStatusInverseKinematicsJointPositions(
                status.handle,
                &raw mut result_body,
                &raw mut result_dof,
                results.as_mut_ptr(),
            )
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to compute inverse kinematics",
                code: success,
            });
        }
        results.truncate(result_dof as usize);
        Ok(results)
    }
}

/// ! =====================================================================================================================================
/// ### Collision Queries & Contact Data
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | getContactPoints | **Implemented** | Returns per-contact data |
/// | getClosestPoints | **Implemented** | Distance queries |
/// | getOverlappingObjects | **Implemented** | AABB queries |
/// | setCollisionFilterPair | **Implemented** | Collision filtering |
/// | setCollisionFilterGroupMask | **Implemented** | Collision filtering |
/// | rayTest | **Implemented** | Single raycast |
/// | rayTestBatch | **Implemented** | Batch raycasts |
impl PhysicsClient {
    pub fn get_contact_points(
        &mut self,
        body_a: Option<i32>,
        body_b: Option<i32>,
        link_a: Option<i32>,
        link_b: Option<i32>,
    ) -> BulletResult<Vec<ContactPoint>> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRequestContactPointInformation(self.handle) };
        unsafe {
            if let Some(body) = body_a {
                ffi::b3SetContactFilterBodyA(command, body);
            }
            if let Some(body) = body_b {
                ffi::b3SetContactFilterBodyB(command, body);
            }
            if let Some(link) = link_a {
                ffi::b3SetContactFilterLinkA(command, link);
            }
            if let Some(link) = link_b {
                ffi::b3SetContactFilterLinkB(command, link);
            }
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CONTACT_POINT_INFORMATION_COMPLETED,
        )?;

        let mut info = MaybeUninit::<ffi::b3ContactInformation>::uninit();
        unsafe {
            ffi::b3GetContactPointInformation(self.handle, info.as_mut_ptr());
            let info = info.assume_init();
            Ok(Self::collect_contact_points(&info))
        }
    }

    pub fn get_closest_points(
        &mut self,
        body_a: i32,
        body_b: i32,
        distance: f64,
        options: &ClosestPointsOptions,
    ) -> BulletResult<Vec<ClosestPoint>> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitClosestDistanceQuery(self.handle) };
        unsafe {
            ffi::b3SetClosestDistanceFilterBodyA(command, body_a);
            ffi::b3SetClosestDistanceFilterBodyB(command, body_b);
            ffi::b3SetClosestDistanceThreshold(command, distance);
            if let Some(link) = options.link_index_a {
                ffi::b3SetClosestDistanceFilterLinkA(command, link);
            }
            if let Some(link) = options.link_index_b {
                ffi::b3SetClosestDistanceFilterLinkB(command, link);
            }
            if let Some(shape) = options.collision_shape_a {
                ffi::b3SetClosestDistanceFilterCollisionShapeA(command, shape);
            }
            if let Some(shape) = options.collision_shape_b {
                ffi::b3SetClosestDistanceFilterCollisionShapeB(command, shape);
            }
            if let Some(position) = options.collision_shape_position_a {
                ffi::b3SetClosestDistanceFilterCollisionShapePositionA(command, position.as_ptr());
            }
            if let Some(position) = options.collision_shape_position_b {
                ffi::b3SetClosestDistanceFilterCollisionShapePositionB(command, position.as_ptr());
            }
            if let Some(orientation) = options.collision_shape_orientation_a {
                ffi::b3SetClosestDistanceFilterCollisionShapeOrientationA(
                    command,
                    orientation.as_ptr(),
                );
            }
            if let Some(orientation) = options.collision_shape_orientation_b {
                ffi::b3SetClosestDistanceFilterCollisionShapeOrientationB(
                    command,
                    orientation.as_ptr(),
                );
            }
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CONTACT_POINT_INFORMATION_COMPLETED,
        )?;

        let mut info = MaybeUninit::<ffi::b3ContactInformation>::uninit();
        unsafe {
            ffi::b3GetClosestPointInformation(self.handle, info.as_mut_ptr());
            let info = info.assume_init();
            Ok(Self::collect_contact_points(&info))
        }
    }

    pub fn get_overlapping_objects(
        &mut self,
        aabb_min: [f64; 3],
        aabb_max: [f64; 3],
    ) -> BulletResult<Vec<OverlappingObject>> {
        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3InitAABBOverlapQuery(self.handle, aabb_min.as_ptr(), aabb_max.as_ptr())
        };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_AABB_OVERLAP_COMPLETED,
        )?;

        let mut data = MaybeUninit::<ffi::b3AABBOverlapData>::uninit();
        unsafe {
            ffi::b3GetAABBOverlapResults(self.handle, data.as_mut_ptr());
            let data = data.assume_init();
            Ok(Self::collect_overlapping_objects(&data))
        }
    }

    pub fn set_collision_filter_pair(
        &mut self,
        body_a: i32,
        body_b: i32,
        link_a: i32,
        link_b: i32,
        enable_collision: bool,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CollisionFilterCommandInit(self.handle) };
        unsafe {
            ffi::b3SetCollisionFilterPair(
                command,
                body_a,
                body_b,
                link_a,
                link_b,
                enable_collision as i32,
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn set_collision_filter_group_mask(
        &mut self,
        body_unique_id: i32,
        link_index: i32,
        collision_filter_group: i32,
        collision_filter_mask: i32,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CollisionFilterCommandInit(self.handle) };
        unsafe {
            ffi::b3SetCollisionFilterGroupMask(
                command,
                body_unique_id,
                link_index,
                collision_filter_group,
                collision_filter_mask,
            );
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn ray_test(
        &mut self,
        ray_from: [f64; 3],
        ray_to: [f64; 3],
        collision_filter_mask: Option<i32>,
        report_hit_number: Option<i32>,
    ) -> BulletResult<Vec<RayHit>> {
        self.ensure_can_submit()?;
        let command = unsafe {
            ffi::b3CreateRaycastCommandInit(
                self.handle,
                ray_from[0],
                ray_from[1],
                ray_from[2],
                ray_to[0],
                ray_to[1],
                ray_to[2],
            )
        };
        unsafe {
            if let Some(mask) = collision_filter_mask {
                ffi::b3RaycastBatchSetCollisionFilterMask(command, mask);
            }
            if let Some(hit_number) = report_hit_number {
                ffi::b3RaycastBatchSetReportHitNumber(command, hit_number);
            }
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_RAY_CAST_INTERSECTIONS_COMPLETED,
        )?;

        let mut info = MaybeUninit::<ffi::b3RaycastInformation>::uninit();
        unsafe {
            ffi::b3GetRaycastInformation(self.handle, info.as_mut_ptr());
            let info = info.assume_init();
            Ok(Self::collect_ray_hits(&info))
        }
    }

    pub fn ray_test_batch(
        &mut self,
        options: &RayTestBatchOptions<'_>,
    ) -> BulletResult<Vec<RayHit>> {
        let from_len = options.ray_from_positions.len();
        let to_len = options.ray_to_positions.len();
        if from_len == 0 || from_len != to_len {
            return Err(BulletError::CommandFailed {
                message: "rayTestBatch requires non-empty rayFrom/rayTo arrays of equal length",
                code: from_len as i32,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreateRaycastBatchCommandInit(self.handle) };
        unsafe {
            if let Some(num_threads) = options.num_threads {
                ffi::b3RaycastBatchSetNumThreads(command, num_threads);
            }
            if let Some(parent_id) = options.parent_object_unique_id {
                ffi::b3RaycastBatchSetParentObject(
                    command,
                    parent_id,
                    options.parent_link_index.unwrap_or(-1),
                );
            }
            if let Some(hit_number) = options.report_hit_number {
                ffi::b3RaycastBatchSetReportHitNumber(command, hit_number);
            }
            if let Some(mask) = options.collision_filter_mask {
                ffi::b3RaycastBatchSetCollisionFilterMask(command, mask);
            }
            if let Some(epsilon) = options.fraction_epsilon {
                ffi::b3RaycastBatchSetFractionEpsilon(command, epsilon);
            }
        }

        for (from, to) in options
            .ray_from_positions
            .iter()
            .zip(options.ray_to_positions.iter())
        {
            unsafe { ffi::b3RaycastBatchAddRay(command, from.as_ptr(), to.as_ptr()) };
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_RAY_CAST_INTERSECTIONS_COMPLETED,
        )?;

        let mut info = MaybeUninit::<ffi::b3RaycastInformation>::uninit();
        unsafe {
            ffi::b3GetRaycastInformation(self.handle, info.as_mut_ptr());
            let info = info.assume_init();
            Ok(Self::collect_ray_hits(&info))
        }
    }

    fn configure_multi_dof_control(
        command: ffi::b3SharedMemoryCommandHandle,
        info: &JointInfo,
        control: &MultiDofControl<'_>,
    ) -> BulletResult<()> {
        let q_index = info.q_index;
        let u_index = info.u_index;
        let q_size = info.q_size.max(0) as usize;
        let u_size = info.u_size.max(0) as usize;

        // 提取通用字段（damping、max_velocity），随后根据模式填充细节。
        let (damping_opt, max_velocity_opt) = match control {
            MultiDofControl::Position { damping, max_velocity, .. }
            | MultiDofControl::Pd { damping, max_velocity, .. }
            | MultiDofControl::StablePd { damping, max_velocity, .. } => (*damping, *max_velocity),
            MultiDofControl::Velocity { damping, .. } | MultiDofControl::Torque { damping, .. } => {
                (*damping, None)
            }
        };

        if let Some(max_velocity) = max_velocity_opt {
            if u_index < 0 || u_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no velocity DOF for max velocity control",
                    code: u_index,
                });
            }
            for dof in 0..u_size {
                unsafe {
                    ffi::b3JointControlSetMaximumVelocity(
                        command,
                        u_index + dof as i32,
                        max_velocity,
                    )
                };
            }
        }
        if let Some(damping) = damping_opt {
            if u_index < 0 || u_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no velocity DOF for damping",
                    code: u_index,
                });
            }
            Self::ensure_slice_len("damping", damping.len(), u_size)?;
            unsafe {
                ffi::b3JointControlSetDampingMultiDof(
                    command,
                    u_index,
                    damping.as_ptr(),
                    u_size as i32,
                );
            }
        }
        if control.is_position_based() {
            if q_index < 0 || q_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no position DOF for position control",
                    code: q_index,
                });
            }
            if u_index < 0 || u_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no velocity DOF for position control",
                    code: u_index,
                });
            }

            // 解构各模式下需要的参数
            let (target_positions, target_velocities, position_gains, velocity_gains, forces) =
                match control {
                    MultiDofControl::Position {
                        target_positions,
                        target_velocities,
                        position_gains,
                        velocity_gains,
                        forces,
                        ..
                    }
                    | MultiDofControl::Pd {
                        target_positions,
                        target_velocities,
                        position_gains,
                        velocity_gains,
                        forces,
                        ..
                    }
                    | MultiDofControl::StablePd {
                        target_positions,
                        target_velocities,
                        position_gains,
                        velocity_gains,
                        forces,
                        ..
                    } => (
                        *target_positions,
                        *target_velocities,
                        *position_gains,
                        *velocity_gains,
                        *forces,
                    ),
                    _ => unreachable!(),
                };

            let positions =
                Self::slice_or_default(target_positions, q_size, 0.0, "targetPositions")?;
            unsafe {
                ffi::b3JointControlSetDesiredPositionMultiDof(
                    command,
                    q_index,
                    positions.as_ptr(),
                    q_size as i32,
                );
            }

            let velocities =
                Self::slice_or_default(target_velocities, u_size, 0.0, "targetVelocities")?;
            unsafe {
                ffi::b3JointControlSetDesiredVelocityMultiDof(
                    command,
                    u_index,
                    velocities.as_ptr(),
                    u_size as i32,
                );
            }

            let kps = Self::slice_or_default(position_gains, u_size, 0.1, "positionGains")?;
            unsafe {
                ffi::b3JointControlSetKpMultiDof(command, u_index, kps.as_ptr(), u_size as i32);
            }

            let kds = Self::slice_or_default(velocity_gains, u_size, 1.0, "velocityGains")?;
            unsafe {
                ffi::b3JointControlSetKdMultiDof(command, u_index, kds.as_ptr(), u_size as i32);
            }

            let forces = Self::slice_or_default(forces, u_size, info.max_force, "forces")?;
            for (dof, &force) in forces.iter().enumerate() {
                unsafe {
                    ffi::b3JointControlSetMaximumForce(command, u_index + dof as i32, force);
                }
            }
        } else if control.is_velocity_based() {
            if u_index < 0 || u_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no velocity DOF for velocity control",
                    code: u_index,
                });
            }

            let (target_velocities, velocity_gains, forces) = match control {
                MultiDofControl::Velocity { target_velocities, velocity_gains, forces, .. } => {
                    (*target_velocities, *velocity_gains, *forces)
                }
                _ => unreachable!(),
            };

            let velocities =
                Self::slice_or_default(target_velocities, u_size, 0.0, "targetVelocities")?;
            unsafe {
                ffi::b3JointControlSetDesiredVelocityMultiDof(
                    command,
                    u_index,
                    velocities.as_ptr(),
                    u_size as i32,
                );
            }

            let kds = Self::slice_or_default(velocity_gains, u_size, 1.0, "velocityGains")?;
            unsafe {
                ffi::b3JointControlSetKdMultiDof(command, u_index, kds.as_ptr(), u_size as i32);
            }

            let forces = Self::slice_or_default(forces, u_size, info.max_force, "forces")?;
            for (dof, &force) in forces.iter().enumerate() {
                unsafe {
                    ffi::b3JointControlSetMaximumForce(command, u_index + dof as i32, force);
                }
            }
        } else if control.is_torque_based() {
            if u_index < 0 || u_size == 0 {
                return Err(BulletError::CommandFailed {
                    message: "Joint has no velocity DOF for torque control",
                    code: u_index,
                });
            }

            let forces = match control {
                MultiDofControl::Torque { forces, .. } => *forces,
                _ => None,
            };
            let forces = Self::slice_or_default(forces, u_size, 0.0, "forces")?;
            unsafe {
                ffi::b3JointControlSetDesiredForceTorqueMultiDof(
                    command,
                    u_index,
                    forces.as_ptr(),
                    u_size as i32,
                );
            }
        }

        Ok(())
    }

    fn ensure_slice_len(name: &'static str, actual: usize, expected: usize) -> BulletResult<()> {
        if expected != actual {
            return Err(BulletError::CommandFailed {
                message: match name {
                    "targetPositions" => "targetPositions length mismatch",
                    "targetVelocities" => "targetVelocities length mismatch",
                    "positionGains" => "positionGains length mismatch",
                    "velocityGains" => "velocityGains length mismatch",
                    "forces" => "forces length mismatch",
                    "damping" => "damping length mismatch",
                    other => {
                        debug_assert!(false, "Unexpected slice name: {other}");
                        "slice length mismatch"
                    }
                },
                code: actual as i32,
            });
        }
        Ok(())
    }

    fn slice_or_default<'a>(
        data: Option<&'a [f64]>,
        expected_len: usize,
        default_value: f64,
        name: &'static str,
    ) -> BulletResult<Cow<'a, [f64]>> {
        if expected_len == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }

        if let Some(slice) = data {
            Self::ensure_slice_len(name, slice.len(), expected_len)?;
            Ok(Cow::Borrowed(slice))
        } else {
            Ok(Cow::Owned(vec![default_value; expected_len]))
        }
    }

    fn collect_contact_points(info: &ffi::b3ContactInformation) -> Vec<ContactPoint> {
        if info.m_contactPointData.is_null() || info.m_numContactPoints <= 0 {
            return Vec::new();
        }
        let count = info.m_numContactPoints as usize;
        let raw = unsafe { slice::from_raw_parts(info.m_contactPointData, count) };
        raw.iter()
            .map(|cp| ContactPoint {
                contact_flags: cp.m_contactFlags,
                body_a: cp.m_bodyUniqueIdA,
                body_b: cp.m_bodyUniqueIdB,
                link_a: cp.m_linkIndexA,
                link_b: cp.m_linkIndexB,
                position_on_a: cp.m_positionOnAInWS,
                position_on_b: cp.m_positionOnBInWS,
                contact_normal_on_b: cp.m_contactNormalOnBInWS,
                contact_distance: cp.m_contactDistance,
                normal_force: cp.m_normalForce,
                linear_friction_force_1: cp.m_linearFrictionForce1,
                linear_friction_direction_1: cp.m_linearFrictionDirection1,
                linear_friction_force_2: cp.m_linearFrictionForce2,
                linear_friction_direction_2: cp.m_linearFrictionDirection2,
            })
            .collect()
    }

    fn collect_overlapping_objects(data: &ffi::b3AABBOverlapData) -> Vec<OverlappingObject> {
        if data.m_overlappingObjects.is_null() || data.m_numOverlappingObjects <= 0 {
            return Vec::new();
        }
        let count = data.m_numOverlappingObjects as usize;
        let raw = unsafe { slice::from_raw_parts(data.m_overlappingObjects, count) };
        raw.iter()
            .map(|obj| OverlappingObject {
                object_unique_id: obj.m_objectUniqueId,
                link_index: obj.m_linkIndex,
            })
            .collect()
    }

    fn collect_ray_hits(info: &ffi::b3RaycastInformation) -> Vec<RayHit> {
        if info.m_rayHits.is_null() || info.m_numRayHits <= 0 {
            return Vec::new();
        }
        let count = info.m_numRayHits as usize;
        let raw = unsafe { slice::from_raw_parts(info.m_rayHits, count) };
        raw.iter()
            .map(|hit| RayHit {
                object_unique_id: hit.m_hitObjectUniqueId,
                link_index: hit.m_hitObjectLinkIndex,
                hit_fraction: hit.m_hitFraction,
                hit_position_world: hit.m_hitPositionWorld,
                hit_normal_world: hit.m_hitNormalWorld,
            })
            .collect()
    }

    fn decode_camera_image(raw: &ffi::b3CameraImageData) -> BulletResult<CameraImage> {
        let width = raw.m_pixel_width;
        let height = raw.m_pixel_height;
        if width <= 0 || height <= 0 {
            return Err(BulletError::CommandFailed {
                message: "Camera returned an empty image",
                code: -1,
            });
        }

        let pixel_count = (width as usize) * (height as usize);
        let mut rgba = Vec::with_capacity(pixel_count * 4);
        if !raw.m_rgb_color_data.is_null() && pixel_count > 0 {
            let data = unsafe { slice::from_raw_parts(raw.m_rgb_color_data, pixel_count * 4) };
            rgba.extend_from_slice(data);
        } else {
            rgba.resize(pixel_count * 4, 0);
        }

        let mut depth = Vec::with_capacity(pixel_count);
        if !raw.m_depth_values.is_null() && pixel_count > 0 {
            let data = unsafe { slice::from_raw_parts(raw.m_depth_values, pixel_count) };
            depth.extend_from_slice(data);
        } else {
            depth.resize(pixel_count, 0.0);
        }

        let mut segmentation_mask = Vec::with_capacity(pixel_count);
        if !raw.m_segmentation_mask_values.is_null() && pixel_count > 0 {
            let data =
                unsafe { slice::from_raw_parts(raw.m_segmentation_mask_values, pixel_count) };
            segmentation_mask.extend(data.iter().copied());
        } else {
            segmentation_mask.resize(pixel_count, -1);
        }

        Ok(CameraImage { width, height, rgba, depth, segmentation_mask })
    }

    fn collect_vr_events(
        data: &ffi::b3VREventsData,
        include_aux_analog_axes: bool,
    ) -> Vec<VrControllerEvent> {
        if data.m_controllerEvents.is_null() || data.m_numControllerEvents <= 0 {
            return Vec::new();
        }
        let count = data.m_numControllerEvents as usize;
        let raw = unsafe { slice::from_raw_parts(data.m_controllerEvents, count) };
        raw.iter()
            .map(|event| {
                let mut position = [0.0_f32; 3];
                position.copy_from_slice(&event.m_pos[..3]);
                let orientation = event.m_orn;
                let mut aux = [0.0_f32; VR_MAX_ANALOG_AXIS * 2];
                if include_aux_analog_axes {
                    aux.copy_from_slice(&event.m_auxAnalogAxis);
                }
                let mut buttons = [0_i32; VR_MAX_BUTTONS];
                buttons.copy_from_slice(&event.m_buttons);
                VrControllerEvent {
                    controller_id: event.m_controllerId,
                    device_type: event.m_deviceType,
                    num_move_events: event.m_numMoveEvents,
                    num_button_events: event.m_numButtonEvents,
                    position,
                    orientation,
                    analog_axis: event.m_analogAxis,
                    aux_analog_axis: aux,
                    buttons,
                }
            })
            .collect()
    }

    fn collect_keyboard_events(data: &ffi::b3KeyboardEventsData) -> Vec<KeyboardEvent> {
        if data.m_keyboardEvents.is_null() || data.m_numKeyboardEvents <= 0 {
            return Vec::new();
        }
        let count = data.m_numKeyboardEvents as usize;
        let raw = unsafe { slice::from_raw_parts(data.m_keyboardEvents, count) };
        raw.iter().copied().map(Into::into).collect()
    }

    fn collect_mouse_events(data: &ffi::b3MouseEventsData) -> Vec<MouseEvent> {
        if data.m_mouseEvents.is_null() || data.m_numMouseEvents <= 0 {
            return Vec::new();
        }
        let count = data.m_numMouseEvents as usize;
        let raw = unsafe { slice::from_raw_parts(data.m_mouseEvents, count) };
        raw.iter().copied().map(Into::into).collect()
    }

    fn collect_visual_shapes(info: &ffi::b3VisualShapeInformation) -> Vec<VisualShapeData> {
        if info.m_visualShapeData.is_null() || info.m_numVisualShapes <= 0 {
            return Vec::new();
        }
        let count = info.m_numVisualShapes as usize;
        let raw = unsafe { slice::from_raw_parts(info.m_visualShapeData, count) };
        raw.iter()
            .map(|shape| {
                let mut frame_position = [0.0_f64; 3];
                frame_position.copy_from_slice(&shape.m_localVisualFrame[..3]);
                let mut frame_orientation = [0.0_f64; 4];
                frame_orientation.copy_from_slice(&shape.m_localVisualFrame[3..7]);
                VisualShapeData {
                    object_unique_id: shape.m_objectUniqueId,
                    link_index: shape.m_linkIndex,
                    geometry_type: shape.m_visualGeometryType,
                    dimensions: shape.m_dimensions,
                    mesh_asset_file_name: Self::read_c_string(&shape.m_meshAssetFileName),
                    local_visual_frame_position: frame_position,
                    local_visual_frame_orientation: frame_orientation,
                    rgba_color: shape.m_rgbaColor,
                    tiny_renderer_texture_id: shape.m_tinyRendererTextureId,
                    texture_unique_id: shape.m_textureUniqueId,
                    opengl_texture_id: shape.m_openglTextureId,
                }
            })
            .collect()
    }

    fn collect_collision_shapes(
        info: &ffi::b3CollisionShapeInformation,
    ) -> Vec<CollisionShapeData> {
        if info.m_collisionShapeData.is_null() || info.m_numCollisionShapes <= 0 {
            return Vec::new();
        }
        let count = info.m_numCollisionShapes as usize;
        let raw = unsafe { slice::from_raw_parts(info.m_collisionShapeData, count) };
        raw.iter()
            .map(|shape| {
                let mut frame_position = [0.0_f64; 3];
                frame_position.copy_from_slice(&shape.m_localCollisionFrame[..3]);
                let mut frame_orientation = [0.0_f64; 4];
                frame_orientation.copy_from_slice(&shape.m_localCollisionFrame[3..7]);
                CollisionShapeData {
                    object_unique_id: shape.m_objectUniqueId,
                    link_index: shape.m_linkIndex,
                    geometry_type: shape.m_collisionGeometryType,
                    dimensions: shape.m_dimensions,
                    mesh_asset_file_name: Self::read_c_string(&shape.m_meshAssetFileName),
                    local_collision_frame_position: frame_position,
                    local_collision_frame_orientation: frame_orientation,
                }
            })
            .collect()
    }

    fn usize_to_i32(value: usize) -> BulletResult<i32> {
        value.try_into().map_err(|_| BulletError::CommandFailed {
            message: "Index exceeds Bullet i32 range",
            code: -1,
        })
    }

    fn path_to_cstring(path: &Path) -> BulletResult<CString> {
        let utf8 = path
            .to_str()
            .ok_or(BulletError::CommandFailed { message: "Path must be valid UTF-8", code: -1 })?;
        Ok(CString::new(utf8)?)
    }
}

/// ! =====================================================================================================================================
/// ### Rendering, Debug & Visualization
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | renderImage | Pending | Obsolete; low priority |
/// | getCameraImage | Implemented | High effort (buffers) |
/// | isNumpyEnabled | Pending | Simple flag query |
/// | computeViewMatrix | Implemented | Math helper |
/// | computeViewMatrixFromYawPitchRoll | Implemented | Math helper |
/// | computeProjectionMatrix | Implemented | Math helper |
/// | computeProjectionMatrixFOV | Implemented | Math helper |
/// | addUserDebugLine | Implemented | Debug draw |
/// | addUserDebugPoints | Implemented | Debug draw |
/// | addUserDebugText | Implemented | Debug draw |
/// | addUserDebugParameter | Implemented | GUI slider/button |
/// | readUserDebugParameter | Implemented | GUI feedback |
/// | removeUserDebugItem | Implemented | Debug cleanup |
/// | removeAllUserDebugItems | Implemented | Debug cleanup |
/// | removeAllUserParameters | Implemented | Debug cleanup |
/// | setDebugObjectColor | Implemented | Debug override |
/// | getDebugVisualizerCamera | Implemented | Visualizer query |
/// | configureDebugVisualizer | Implemented | Visualizer toggle |
/// | resetDebugVisualizerCamera | Implemented | Visualizer camera |
/// | getVisualShapeData | Implemented | Visual query |
/// | getCollisionShapeData | Implemented | Collision query |
impl PhysicsClient {
    pub fn get_camera_image(&mut self, request: &CameraImageOptions) -> BulletResult<CameraImage> {
        if request.width <= 0 || request.height <= 0 {
            return Err(BulletError::CommandFailed {
                message: "Camera image width/height must be positive",
                code: -1,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRequestCameraImage(self.handle) };
        unsafe {
            ffi::b3RequestCameraImageSetPixelResolution(command, request.width, request.height);
        }

        if let (Some(mut view), Some(mut projection)) =
            (request.view_matrix, request.projection_matrix)
        {
            unsafe {
                ffi::b3RequestCameraImageSetCameraMatrices(
                    command,
                    view.as_mut_ptr(),
                    projection.as_mut_ptr(),
                );
            }
        }
        if let Some(mut direction) = request.light_direction {
            unsafe { ffi::b3RequestCameraImageSetLightDirection(command, direction.as_mut_ptr()) };
        }
        if let Some(mut color) = request.light_color {
            unsafe { ffi::b3RequestCameraImageSetLightColor(command, color.as_mut_ptr()) };
        }
        if let Some(distance) = request.light_distance
            && distance >= 0.0
        {
            unsafe { ffi::b3RequestCameraImageSetLightDistance(command, distance) };
        }
        if let Some(has_shadow) = request.shadow {
            unsafe { ffi::b3RequestCameraImageSetShadow(command, has_shadow as i32) };
        }
        if let Some(coeff) = request.light_ambient_coeff
            && coeff >= 0.0
        {
            unsafe { ffi::b3RequestCameraImageSetLightAmbientCoeff(command, coeff) };
        }
        if let Some(coeff) = request.light_diffuse_coeff
            && coeff >= 0.0
        {
            unsafe { ffi::b3RequestCameraImageSetLightDiffuseCoeff(command, coeff) };
        }
        if let Some(coeff) = request.light_specular_coeff
            && coeff >= 0.0
        {
            unsafe { ffi::b3RequestCameraImageSetLightSpecularCoeff(command, coeff) };
        }
        if let Some(renderer) = request.renderer {
            unsafe { ffi::b3RequestCameraImageSelectRenderer(command, renderer as i32) };
        }
        if let Some(flags) = request.flags {
            unsafe { ffi::b3RequestCameraImageSetFlags(command, flags) };
        }
        if let (Some(mut view), Some(mut projection)) = (
            request.projective_texture_view,
            request.projective_texture_projection,
        ) {
            unsafe {
                ffi::b3RequestCameraImageSetProjectiveTextureMatrices(
                    command,
                    view.as_mut_ptr(),
                    projection.as_mut_ptr(),
                );
            }
        }

        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CAMERA_IMAGE_COMPLETED,
        )?;

        let mut raw = ffi::b3CameraImageData::default();
        unsafe { ffi::b3GetCameraImageData(self.handle, &raw mut raw) };
        Self::decode_camera_image(&raw)
    }

    pub fn compute_view_matrix(
        camera_eye_position: [f32; 3],
        camera_target_position: [f32; 3],
        camera_up_vector: [f32; 3],
    ) -> [f32; 16] {
        let mut matrix = [0.0_f32; 16];
        unsafe {
            ffi::b3ComputeViewMatrixFromPositions(
                camera_eye_position.as_ptr(),
                camera_target_position.as_ptr(),
                camera_up_vector.as_ptr(),
                matrix.as_mut_ptr(),
            );
        }
        matrix
    }

    pub fn compute_view_matrix_from_yaw_pitch_roll(
        camera_target_position: [f32; 3],
        distance: f32,
        yaw: f32,
        pitch: f32,
        roll: f32,
        up_axis: i32,
    ) -> [f32; 16] {
        let mut matrix = [0.0_f32; 16];
        unsafe {
            ffi::b3ComputeViewMatrixFromYawPitchRoll(
                camera_target_position.as_ptr(),
                distance,
                yaw,
                pitch,
                roll,
                up_axis,
                matrix.as_mut_ptr(),
            );
        }
        matrix
    }

    pub fn compute_projection_matrix(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near_val: f32,
        far_val: f32,
    ) -> [f32; 16] {
        let mut matrix = [0.0_f32; 16];
        unsafe {
            ffi::b3ComputeProjectionMatrix(
                left,
                right,
                bottom,
                top,
                near_val,
                far_val,
                matrix.as_mut_ptr(),
            );
        }
        matrix
    }

    pub fn compute_projection_matrix_fov(
        fov: f32,
        aspect: f32,
        near_val: f32,
        far_val: f32,
    ) -> [f32; 16] {
        let mut matrix = [0.0_f32; 16];
        unsafe {
            ffi::b3ComputeProjectionMatrixFOV(fov, aspect, near_val, far_val, matrix.as_mut_ptr());
        }
        matrix
    }

    pub fn add_user_debug_line(&mut self, options: &DebugLineOptions) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let color = options.color.unwrap_or([1.0; 3]);
        let command = unsafe {
            ffi::b3InitUserDebugDrawAddLine3D(
                self.handle,
                options.from.as_ptr(),
                options.to.as_ptr(),
                color.as_ptr(),
                options.line_width,
                options.life_time,
            )
        };

        if let Some(parent_id) = options.parent_object_unique_id {
            let link_index = options.parent_link_index.unwrap_or(-1);
            unsafe { ffi::b3UserDebugItemSetParentObject(command, parent_id, link_index) };
        }
        if let Some(replace_id) = options.replace_item_unique_id {
            unsafe { ffi::b3UserDebugItemSetReplaceItemUniqueId(command, replace_id) };
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_DEBUG_DRAW_COMPLETED,
        )?;
        let unique_id = unsafe { ffi::b3GetDebugItemUniqueId(status.handle) };
        Ok(unique_id)
    }

    pub fn add_user_debug_points(&mut self, options: &DebugPointsOptions<'_>) -> BulletResult<i32> {
        if options.positions.len() != options.colors.len() {
            return Err(BulletError::CommandFailed {
                message: "pointColors length must match pointPositions",
                code: options.colors.len() as i32,
            });
        }
        if options.positions.is_empty() {
            return Err(BulletError::CommandFailed {
                message: "At least one point is required",
                code: 0,
            });
        }

        self.ensure_can_submit()?;
        let positions = Self::flatten_points(options.positions);
        let colors = Self::flatten_points(options.colors);
        let command = unsafe {
            ffi::b3InitUserDebugDrawAddPoints3D(
                self.handle,
                positions.as_ptr(),
                colors.as_ptr(),
                options.point_size,
                options.life_time,
                options.positions.len() as i32,
            )
        };

        if let Some(parent_id) = options.parent_object_unique_id {
            let link_index = options.parent_link_index.unwrap_or(-1);
            unsafe { ffi::b3UserDebugItemSetParentObject(command, parent_id, link_index) };
        }
        if let Some(replace_id) = options.replace_item_unique_id {
            unsafe { ffi::b3UserDebugItemSetReplaceItemUniqueId(command, replace_id) };
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_DEBUG_DRAW_COMPLETED,
        )?;
        let unique_id = unsafe { ffi::b3GetDebugItemUniqueId(status.handle) };
        Ok(unique_id)
    }

    pub fn add_user_debug_text(&mut self, options: &DebugTextOptions<'_>) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let color = options.color.unwrap_or([1.0; 3]);
        let text = CString::new(options.text)?;
        let command = unsafe {
            ffi::b3InitUserDebugDrawAddText3D(
                self.handle,
                text.as_ptr(),
                options.position.as_ptr(),
                color.as_ptr(),
                options.size,
                options.life_time,
            )
        };

        if let Some(parent_id) = options.parent_object_unique_id {
            let link_index = options.parent_link_index.unwrap_or(-1);
            unsafe { ffi::b3UserDebugItemSetParentObject(command, parent_id, link_index) };
        }
        if let Some(orientation) = options.orientation {
            const USER_DEBUG_HAS_TEXT_ORIENTATION: i32 = 512;
            unsafe {
                ffi::b3UserDebugTextSetOptionFlags(command, USER_DEBUG_HAS_TEXT_ORIENTATION);
                ffi::b3UserDebugTextSetOrientation(command, orientation.as_ptr());
            }
        }
        if let Some(replace_id) = options.replace_item_unique_id {
            unsafe { ffi::b3UserDebugItemSetReplaceItemUniqueId(command, replace_id) };
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_DEBUG_DRAW_COMPLETED,
        )?;
        let unique_id = unsafe { ffi::b3GetDebugItemUniqueId(status.handle) };
        Ok(unique_id)
    }

    pub fn add_user_debug_parameter(
        &mut self,
        options: &DebugParameterOptions<'_>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let name = CString::new(options.name)?;
        let command = unsafe {
            ffi::b3InitUserDebugAddParameter(
                self.handle,
                name.as_ptr(),
                options.range_min,
                options.range_max,
                options.start_value,
            )
        };

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_DEBUG_DRAW_PARAMETER_COMPLETED,
        )?;
        let unique_id = unsafe { ffi::b3GetDebugItemUniqueId(status.handle) };
        Ok(unique_id)
    }

    pub fn read_user_debug_parameter(&mut self, parameter_id: i32) -> BulletResult<f64> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitUserDebugReadParameter(self.handle, parameter_id) };
        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_USER_DEBUG_DRAW_PARAMETER_COMPLETED,
        )?;

        let mut value = 0.0_f64;
        let success = unsafe { ffi::b3GetStatusDebugParameterValue(status.handle, &raw mut value) };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Unable to read user debug parameter",
                code: success,
            });
        }
        Ok(value)
    }

    pub fn remove_user_debug_item(&mut self, item_unique_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitUserDebugDrawRemove(self.handle, item_unique_id) };
        let _ = self.submit_command(command)?;
        Ok(())
    }

    pub fn remove_all_user_debug_items(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitUserDebugDrawRemoveAll(self.handle) };
        let _ = self.submit_command(command)?;
        Ok(())
    }

    pub fn remove_all_user_parameters(&mut self) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitUserRemoveAllParameters(self.handle) };
        let _ = self.submit_command(command)?;
        Ok(())
    }

    pub fn set_debug_object_color(
        &mut self,
        object_unique_id: i32,
        link_index: i32,
        color: Option<[f64; 3]>,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitDebugDrawingCommand(self.handle) };
        unsafe {
            if let Some(color) = color {
                ffi::b3SetDebugObjectColor(command, object_unique_id, link_index, color.as_ptr());
            } else {
                ffi::b3RemoveDebugObjectColor(command, object_unique_id, link_index);
            }
        }
        let _ = self.submit_command(command)?;
        Ok(self)
    }

    pub fn get_debug_visualizer_camera(&mut self) -> BulletResult<DebugVisualizerCamera> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitRequestOpenGLVisualizerCameraCommand(self.handle) };
        let status = self.submit_command(command)?;

        let mut camera = ffi::b3OpenGLVisualizerCameraInfo::default();
        let success =
            unsafe { ffi::b3GetStatusOpenGLVisualizerCamera(status.handle, &raw mut camera) };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Unable to fetch debug visualizer camera",
                code: success,
            });
        }

        Ok(DebugVisualizerCamera {
            width: camera.m_width,
            height: camera.m_height,
            view_matrix: camera.m_viewMatrix,
            projection_matrix: camera.m_projectionMatrix,
            camera_up: camera.m_camUp,
            camera_forward: camera.m_camForward,
            horizontal: camera.m_horizontal,
            vertical: camera.m_vertical,
            yaw: camera.m_yaw,
            pitch: camera.m_pitch,
            distance: camera.m_dist,
            target: camera.m_target,
        })
    }

    pub fn configure_debug_visualizer(
        &mut self,
        options: &DebugVisualizerOptions,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitConfigureOpenGLVisualizer(self.handle) };

        match options {
            DebugVisualizerOptions::Flag(flag, enabled) => unsafe {
                ffi::b3ConfigureOpenGLVisualizerSetVisualizationFlags(
                    command,
                    *flag as i32,
                    *enabled as i32,
                );
            },
            DebugVisualizerOptions::LightPosition(position) => unsafe {
                let mut pos_copy = *position;
                ffi::b3ConfigureOpenGLVisualizerSetLightPosition(command, pos_copy.as_mut_ptr());
            },
            DebugVisualizerOptions::ShadowMapResolution(resolution) => unsafe {
                ffi::b3ConfigureOpenGLVisualizerSetShadowMapResolution(command, *resolution);
            },
            DebugVisualizerOptions::ShadowMapWorldSize(world_size) => unsafe {
                ffi::b3ConfigureOpenGLVisualizerSetShadowMapWorldSize(command, *world_size);
            },
            DebugVisualizerOptions::RemoteSyncTransformInterval(interval) => unsafe {
                ffi::b3ConfigureOpenGLVisualizerSetRemoteSyncTransformInterval(command, *interval);
            },
            DebugVisualizerOptions::ShadowMapIntensity(intensity) => unsafe {
                ffi::b3ConfigureOpenGLVisualizerSetShadowMapIntensity(command, *intensity);
            },
            DebugVisualizerOptions::RgbBackground(rgb) => unsafe {
                let mut rgb_copy = *rgb;
                ffi::b3ConfigureOpenGLVisualizerSetLightRgbBackground(
                    command,
                    rgb_copy.as_mut_ptr(),
                );
            },
        }

        let _ = self.submit_command(command)?;
        Ok(self)
    }

    pub fn reset_debug_visualizer_camera(
        &mut self,
        options: &ResetDebugVisualizerCameraOptions,
    ) -> BulletResult<&mut Self> {
        if options.distance < 0.0 {
            return Err(BulletError::CommandFailed {
                message: "cameraDistance must be non-negative",
                code: -1,
            });
        }

        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3InitConfigureOpenGLVisualizer(self.handle) };
        unsafe {
            ffi::b3ConfigureOpenGLVisualizerSetViewMatrix(
                command,
                options.distance,
                options.pitch,
                options.yaw,
                options.target.as_ptr(),
            );
        }
        let _ = self.submit_command(command)?;
        Ok(self)
    }

    pub fn get_visual_shape_data(
        &mut self,
        object_unique_id: i32,
    ) -> BulletResult<Vec<VisualShapeData>> {
        self.ensure_can_submit()?;
        let command =
            unsafe { ffi::b3InitRequestVisualShapeInformation(self.handle, object_unique_id) };
        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_VISUAL_SHAPE_INFO_COMPLETED,
        )?;
        let mut info = ffi::b3VisualShapeInformation::default();
        unsafe { ffi::b3GetVisualShapeInformation(self.handle, &raw mut info) };
        Ok(Self::collect_visual_shapes(&info))
    }

    pub fn get_collision_shape_data(
        &mut self,
        object_unique_id: i32,
        link_index: Option<i32>,
    ) -> BulletResult<Vec<CollisionShapeData>> {
        self.ensure_can_submit()?;
        let link_index = link_index.unwrap_or(-1);
        let command = unsafe {
            ffi::b3InitRequestCollisionShapeInformation(self.handle, object_unique_id, link_index)
        };
        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_COLLISION_SHAPE_INFO_COMPLETED,
        )?;
        let mut info = ffi::b3CollisionShapeInformation::default();
        unsafe { ffi::b3GetCollisionShapeInformation(self.handle, &raw mut info) };
        Ok(Self::collect_collision_shapes(&info))
    }
}

/// ! =====================================================================================================================================
/// ### Math & Transform Utilities
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | getQuaternionFromEuler | Pending | Utility |
/// | getEulerFromQuaternion | Pending | Utility |
/// | multiplyTransforms | Pending | Utility |
/// | invertTransform | Pending | Utility |
/// | getMatrixFromQuaternion | Pending | Utility |
/// | getQuaternionSlerp | Pending | Utility |
/// | getQuaternionFromAxisAngle | Pending | Utility |
/// | getAxisAngleFromQuaternion | Pending | Utility |
/// | getDifferenceQuaternion | Pending | Utility |
/// | getAxisDifferenceQuaternion | Pending | Utility |
/// | calculateVelocityQuaternion | Pending | Utility |
/// | rotateVector | Pending | Utility |
impl PhysicsClient {
    fn _math_transform_utilities() {}
}

/// ! =====================================================================================================================================
/// ### VR, Input, Logging, Plugins & Misc
///
/// | API | Status | Notes |
/// | --- | --- | --- |
/// | getVREvents | Implemented | VR input |
/// | setVRCameraState | Implemented | VR camera |
/// | getKeyboardEvents | Implemented | Input polling |
/// | getMouseEvents | Implemented | Input polling |
/// | startStateLogging | Implemented | Logging |
/// | stopStateLogging | Implemented | Logging |
/// | loadPlugin | Implemented | Plugin system |
/// | unloadPlugin | Implemented | Plugin system |
/// | executePluginCommand | Implemented | Plugin system |
/// | submitProfileTiming | Implemented | Profiling |
/// | setTimeOut | Implemented | Client timeout |
/// | getAPIVersion | Implemented | Static constant |
impl PhysicsClient {
    pub fn get_vr_events(
        &mut self,
        options: &VrEventsOptions,
    ) -> BulletResult<Vec<VrControllerEvent>> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3RequestVREventsCommandInit(self.handle) };
        let filter = options
            .device_type_filter
            .unwrap_or(ffi::VR_DEVICE_CONTROLLER);
        unsafe {
            ffi::b3VREventsSetDeviceTypeFilter(command, filter);
        }

        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_VR_EVENTS_DATA_COMPLETED,
        )?;

        let mut raw = ffi::b3VREventsData::default();
        unsafe { ffi::b3GetVREventsData(self.handle, &raw mut raw) };
        Ok(Self::collect_vr_events(
            &raw,
            options.include_aux_analog_axes,
        ))
    }

    pub fn set_vr_camera_state(
        &mut self,
        options: &VrCameraStateOptions,
    ) -> BulletResult<&mut Self> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3SetVRCameraStateCommandInit(self.handle) };

        if let Some(position) = options.root_position {
            unsafe { ffi::b3SetVRCameraRootPosition(command, position.as_ptr()) };
        }
        if let Some(orientation) = options.root_orientation {
            unsafe { ffi::b3SetVRCameraRootOrientation(command, orientation.as_ptr()) };
        }
        if let Some(object_id) = options.tracking_object_unique_id {
            unsafe { ffi::b3SetVRCameraTrackingObject(command, object_id) };
        }
        if let Some(flag) = options.tracking_flag {
            unsafe { ffi::b3SetVRCameraTrackingObjectFlag(command, flag) };
        }

        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn get_keyboard_events(&mut self) -> BulletResult<Vec<KeyboardEvent>> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3RequestKeyboardEventsCommandInit(self.handle) };
        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_KEYBOARD_EVENTS_DATA_COMPLETED,
        )?;

        let mut raw = ffi::b3KeyboardEventsData::default();
        unsafe { ffi::b3GetKeyboardEventsData(self.handle, &raw mut raw) };
        Ok(Self::collect_keyboard_events(&raw))
    }

    pub fn get_mouse_events(&mut self) -> BulletResult<Vec<MouseEvent>> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3RequestMouseEventsCommandInit(self.handle) };
        let _status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_REQUEST_MOUSE_EVENTS_DATA_COMPLETED,
        )?;

        let mut raw = ffi::b3MouseEventsData::default();
        unsafe { ffi::b3GetMouseEventsData(self.handle, &raw mut raw) };
        Ok(Self::collect_mouse_events(&raw))
    }

    pub fn start_state_logging(
        &mut self,
        logging_type: LoggingType,
        file_name: impl AsRef<Path>,
        options: Option<impl Into<StateLoggingOptions>>,
    ) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let file_c = Self::path_to_cstring(file_name.as_ref())?;
        let command = unsafe { ffi::b3StateLoggingCommandInit(self.handle) };
        unsafe { ffi::b3StateLoggingStart(command, logging_type as i32, file_c.as_ptr()) };

        let options = options.map(Into::into).unwrap_or_default();
        if let Some(object_ids) = options.object_unique_ids {
            for object_id in object_ids {
                unsafe { ffi::b3StateLoggingAddLoggingObjectUniqueId(command, object_id) };
            }
        }
        if let Some(max_log_dof) = options.max_log_dof {
            unsafe { ffi::b3StateLoggingSetMaxLogDof(command, max_log_dof) };
        }
        if let Some(body_a) = options.body_unique_id_a {
            unsafe { ffi::b3StateLoggingSetBodyAUniqueId(command, body_a) };
        }
        if let Some(body_b) = options.body_unique_id_b {
            unsafe { ffi::b3StateLoggingSetBodyBUniqueId(command, body_b) };
        }
        if let Some(link_a) = options.link_index_a {
            unsafe { ffi::b3StateLoggingSetLinkIndexA(command, link_a) };
        }
        if let Some(link_b) = options.link_index_b {
            unsafe { ffi::b3StateLoggingSetLinkIndexB(command, link_b) };
        }
        if let Some(device_filter) = options.device_type_filter {
            unsafe { ffi::b3StateLoggingSetDeviceTypeFilter(command, device_filter) };
        }
        if let Some(flags) = options.log_flags {
            unsafe { ffi::b3StateLoggingSetLogFlags(command, flags.bits()) };
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_STATE_LOGGING_START_COMPLETED,
        )?;

        let logging_id = unsafe { ffi::b3GetStatusLoggingUniqueId(status.handle) };
        if logging_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to start state logging",
                code: logging_id,
            });
        }
        Ok(logging_id)
    }

    pub fn stop_state_logging(&mut self, logging_id: i32) -> BulletResult<&mut Self> {
        if logging_id < 0 {
            return Ok(self);
        }
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3StateLoggingCommandInit(self.handle) };
        unsafe { ffi::b3StateLoggingStop(command, logging_id) };
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_STATE_LOGGING_COMPLETED,
        )?;
        Ok(self)
    }

    pub fn load_plugin(&mut self, plugin_path: &str, postfix: Option<&str>) -> BulletResult<i32> {
        self.ensure_can_submit()?;
        let path_c = CString::new(plugin_path)?;
        let postfix_c = postfix.map(CString::new).transpose()?;
        let command = unsafe { ffi::b3CreateCustomCommand(self.handle) };
        unsafe { ffi::b3CustomCommandLoadPlugin(command, path_c.as_ptr()) };
        if let Some(ref postfix_c) = postfix_c {
            unsafe { ffi::b3CustomCommandLoadPluginSetPostFix(command, postfix_c.as_ptr()) };
        }

        let status = self.submit_command(command)?;
        let plugin_id = unsafe { ffi::b3GetStatusPluginUniqueId(status.handle) };
        if plugin_id < 0 {
            return Err(BulletError::CommandFailed {
                message: "Bullet failed to load plugin",
                code: plugin_id,
            });
        }
        Ok(plugin_id)
    }

    pub fn unload_plugin(&mut self, plugin_unique_id: i32) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreateCustomCommand(self.handle) };
        unsafe { ffi::b3CustomCommandUnloadPlugin(command, plugin_unique_id) };
        self.submit_command(command)?;
        Ok(())
    }

    pub fn execute_plugin_command(
        &mut self,
        options: &PluginCommandOptions<'_>,
    ) -> BulletResult<PluginCommandResult> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3CreateCustomCommand(self.handle) };
        let text_c = options.text_argument.map(CString::new).transpose()?;
        let text_ptr = text_c.as_ref().map_or(ptr::null(), |value| value.as_ptr());
        unsafe {
            ffi::b3CustomCommandExecutePluginCommand(command, options.plugin_unique_id, text_ptr);
        }

        if let Some(int_args) = options.int_args {
            for &value in int_args {
                unsafe { ffi::b3CustomCommandExecuteAddIntArgument(command, value) };
            }
        }
        if let Some(float_args) = options.float_args {
            for &value in float_args {
                unsafe { ffi::b3CustomCommandExecuteAddFloatArgument(command, value) };
            }
        }

        let status = self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CUSTOM_COMMAND_COMPLETED,
        )?;

        let status_code = unsafe { ffi::b3GetStatusPluginCommandResult(status.handle) };
        let mut raw = ffi::b3UserDataValue { m_type: 0, m_length: 0, m_data1: ptr::null() };
        let has_data =
            unsafe { ffi::b3GetStatusPluginCommandReturnData(self.handle, &raw mut raw) } != 0;
        let return_data = if has_data && raw.m_length > 0 && !raw.m_data1.is_null() {
            let len = raw.m_length as usize;
            let bytes = unsafe { slice::from_raw_parts(raw.m_data1, len) };
            let data = bytes.iter().map(|&b| b as i32).collect();
            Some(PluginCommandReturnData { value_type: raw.m_type, data })
        } else {
            None
        };

        Ok(PluginCommandResult { status: status_code, return_data })
    }

    pub fn submit_profile_timing(&mut self, event_name: Option<&str>) -> BulletResult<()> {
        self.ensure_can_submit()?;
        let name_c = event_name.map(CString::new).transpose()?;
        let command = unsafe {
            ffi::b3ProfileTimingCommandInit(
                self.handle,
                name_c.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            )
        };
        unsafe {
            ffi::b3SetProfileTimingType(command, name_c.is_some().into());
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_CLIENT_COMMAND_COMPLETED,
        )?;
        Ok(())
    }

    pub fn set_time_out(&mut self, time_out_seconds: f64) -> BulletResult<&mut Self> {
        if time_out_seconds < 0.0 {
            return Err(BulletError::CommandFailed {
                message: "Timeout must be non-negative",
                code: -1,
            });
        }
        self.ensure_can_submit()?;
        unsafe { ffi::b3SetTimeOut(self.handle, time_out_seconds) };
        Ok(self)
    }

    pub fn get_api_version(&self) -> i32 {
        ffi::SHARED_MEMORY_MAGIC_NUMBER
    }
}

/// ! =====================================================================================================================================
/// ### tool functions
impl PhysicsClient {
    fn ensure_can_submit(&mut self) -> BulletResult<()> {
        if self.can_submit_command() {
            Ok(())
        } else {
            Err(BulletError::ServerUnavailable(
                "Physics server is not running",
            ))
        }
    }

    fn apply_physics_engine_update(
        command: ffi::b3SharedMemoryCommandHandle,
        update: &PhysicsEngineParametersUpdate,
    ) {
        unsafe {
            if let Some(num_solver_iterations) = update.num_solver_iterations {
                ffi::b3PhysicsParamSetNumSolverIterations(command, num_solver_iterations as i32);
            }
            if let Some(minimum_solver_island_size) = update.minimum_solver_island_size {
                ffi::b3PhysicsParameterSetMinimumSolverIslandSize(
                    command,
                    minimum_solver_island_size as i32,
                );
            }
            if let Some(solver_residual_threshold) = update.solver_residual_threshold {
                assert!(solver_residual_threshold.is_sign_positive());
                ffi::b3PhysicsParamSetSolverResidualThreshold(command, solver_residual_threshold);
            }
            if let Some(collision_filter_mode) = update.collision_filter_mode {
                ffi::b3PhysicsParamSetCollisionFilterMode(command, collision_filter_mode as i32);
            }
            if let Some(num_sub_steps) = update.num_sub_steps {
                ffi::b3PhysicsParamSetNumSubSteps(command, num_sub_steps as i32);
            }
            if let Some(fixed_time_step) = update.fixed_time_step {
                ffi::b3PhysicsParamSetTimeStep(command, fixed_time_step.as_secs_f64());
            }
            if let Some(use_split_impulse) = update.use_split_impulse {
                if use_split_impulse {
                    ffi::b3PhysicsParamSetUseSplitImpulse(command, 1);
                } else {
                    ffi::b3PhysicsParamSetUseSplitImpulse(command, 0);
                }
            }
            if let Some(split_impulse_penetration_threshold) =
                update.split_impulse_penetration_threshold
            {
                assert!(split_impulse_penetration_threshold.is_sign_positive());
                ffi::b3PhysicsParamSetSplitImpulsePenetrationThreshold(
                    command,
                    split_impulse_penetration_threshold,
                );
            }
            if let Some(contact_breaking_threshold) = update.contact_breaking_threshold {
                assert!(contact_breaking_threshold.is_sign_positive());
                ffi::b3PhysicsParamSetContactBreakingThreshold(command, contact_breaking_threshold);
            }
            if let Some(contact_slop) = update.contact_slop {
                assert!(contact_slop.is_sign_positive());
                ffi::b3PhysicsParamSetContactSlop(command, contact_slop);
            }
            if let Some(max_num_cmd_per_1_ms) = update.max_num_cmd_per_1_ms {
                assert!(max_num_cmd_per_1_ms >= -1);
                ffi::b3PhysicsParamSetMaxNumCommandsPer1ms(command, max_num_cmd_per_1_ms);
            }
            if let Some(restitution_velocity_threshold) = update.restitution_velocity_threshold {
                assert!(restitution_velocity_threshold.is_sign_positive());
                ffi::b3PhysicsParamSetRestitutionVelocityThreshold(
                    command,
                    restitution_velocity_threshold,
                );
            }
            if let Some(enable_file_caching) = update.enable_file_caching {
                if enable_file_caching {
                    ffi::b3PhysicsParamSetEnableFileCaching(command, 1);
                } else {
                    ffi::b3PhysicsParamSetEnableFileCaching(command, 0);
                }
            }
            if let Some(erp) = update.erp {
                assert!(erp.is_sign_positive());
                ffi::b3PhysicsParamSetDefaultNonContactERP(command, erp);
            }
            if let Some(contact_erp) = update.contact_erp {
                assert!(contact_erp.is_sign_positive());
                ffi::b3PhysicsParamSetDefaultContactERP(command, contact_erp);
            }
            if let Some(friction_erp) = update.friction_erp {
                assert!(friction_erp.is_sign_positive());
                ffi::b3PhysicsParamSetDefaultFrictionERP(command, friction_erp);
            }
            if let Some(enable_cone_friction) = update.enable_cone_friction {
                if enable_cone_friction {
                    ffi::b3PhysicsParamSetEnableConeFriction(command, 1);
                } else {
                    ffi::b3PhysicsParamSetEnableConeFriction(command, 0);
                }
            }
            if let Some(deterministic_overlapping_pairs) = update.deterministic_overlapping_pairs {
                if deterministic_overlapping_pairs {
                    ffi::b3PhysicsParameterSetDeterministicOverlappingPairs(command, 1);
                } else {
                    ffi::b3PhysicsParameterSetDeterministicOverlappingPairs(command, 0);
                }
            }
            if let Some(allowed_ccd_penetration) = update.allowed_ccd_penetration {
                assert!(allowed_ccd_penetration.is_sign_positive());
                ffi::b3PhysicsParameterSetAllowedCcdPenetration(command, allowed_ccd_penetration);
            }
            if let Some(joint_feedback_mode) = update.joint_feedback_mode {
                ffi::b3PhysicsParameterSetJointFeedbackMode(command, joint_feedback_mode as i32);
            }
            if let Some(enable_sat) = update.enable_sat {
                if enable_sat {
                    ffi::b3PhysicsParameterSetEnableSAT(command, 1);
                } else {
                    ffi::b3PhysicsParameterSetEnableSAT(command, 0);
                }
            }
            if let Some(constraint_solver_type) = update.constraint_solver_type {
                let val = constraint_solver_type as i32;
                println!("{val:?}");
                ffi::b3PhysicsParameterSetConstraintSolverType(command, val);
            }
            if let Some(global_cfm) = update.global_cfm {
                assert!(global_cfm.is_sign_positive());
                ffi::b3PhysicsParamSetDefaultGlobalCFM(command, global_cfm);
            }
            if let Some(report_solver_analytics) = update.report_solver_analytics {
                if report_solver_analytics {
                    ffi::b3PhysicsParamSetSolverAnalytics(command, 1);
                } else {
                    ffi::b3PhysicsParamSetSolverAnalytics(command, 0);
                }
            }
            if let Some(warm_starting_factor) = update.warm_starting_factor {
                assert!(warm_starting_factor.is_sign_positive());
                ffi::b3PhysicsParamSetWarmStartingFactor(command, warm_starting_factor);
            }
            if let Some(sparse_sdf_voxel_size) = update.sparse_sdf_voxel_size {
                assert!(sparse_sdf_voxel_size.is_sign_positive());
                ffi::b3PhysicsParameterSetSparseSdfVoxelSize(command, sparse_sdf_voxel_size);
            }
            if let Some(num_non_contact_inner_iterations) = update.num_non_contact_inner_iterations
            {
                assert!(num_non_contact_inner_iterations >= 1);
                ffi::b3PhysicsParamSetNumNonContactInnerIterations(
                    command,
                    num_non_contact_inner_iterations as i32,
                );
            }
        }
    }

    fn submit_command(
        &mut self,
        command: ffi::b3SharedMemoryCommandHandle,
    ) -> BulletResult<CommandStatus> {
        unsafe {
            let handle = ffi::b3SubmitClientCommandAndWaitStatus(self.handle, command);
            if handle.is_null() {
                return Err(BulletError::NullPointer(
                    "Bullet returned a null status handle",
                ));
            }
            let status_type = ffi::b3GetStatusType(handle);
            Ok(CommandStatus { handle, status_type })
        }
    }

    fn submit_simple_command(
        &mut self,
        command: ffi::b3SharedMemoryCommandHandle,
        expected_status: ffi::EnumSharedMemoryServerStatus,
    ) -> BulletResult<CommandStatus> {
        let status = self.submit_command(command)?;
        let expected = expected_status as i32;
        if status.status_type != expected {
            return Err(BulletError::UnexpectedStatus { expected, actual: status.status_type });
        }
        Ok(status)
    }

    fn collect_body_indices(
        status_handle: ffi::b3SharedMemoryStatusHandle,
    ) -> BulletResult<Vec<i32>> {
        if status_handle.is_null() {
            return Err(BulletError::NullPointer(
                "Bullet returned a null status handle while collecting body indices",
            ));
        }

        let capacity = usize::try_from(ffi::MAX_SDF_BODIES).unwrap_or(512).max(1);
        let mut buffer = vec![0_i32; capacity];
        let count = unsafe {
            ffi::b3GetStatusBodyIndices(status_handle, buffer.as_mut_ptr(), capacity as i32)
        };
        if count < 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to read body indices from status",
                code: count,
            });
        }
        buffer.truncate(count as usize);
        Ok(buffer)
    }

    fn add_collision_geometry(
        &self,
        command: ffi::b3SharedMemoryCommandHandle,
        geometry: &CollisionGeometry<'_>,
        scratch: &mut GeometryScratch,
    ) -> BulletResult<i32> {
        use CollisionGeometry::*;

        let index = match geometry {
            Sphere { radius } => {
                if *radius <= 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Sphere radius must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateCollisionShapeAddSphere(command, *radius) }
            }
            Box { half_extents } => unsafe {
                ffi::b3CreateCollisionShapeAddBox(command, half_extents.as_ptr())
            },
            Capsule { radius, height } => {
                if *radius <= 0.0 || *height < 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Capsule dimensions must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateCollisionShapeAddCapsule(command, *radius, *height) }
            }
            Cylinder { radius, height } => {
                if *radius <= 0.0 || *height <= 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Cylinder dimensions must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateCollisionShapeAddCylinder(command, *radius, *height) }
            }
            Plane { normal, constant } => unsafe {
                ffi::b3CreateCollisionShapeAddPlane(command, normal.as_ptr(), *constant)
            },
            &MeshFile { file, scale } => {
                let file_ptr = scratch.push_c_string(file)?;
                unsafe { ffi::b3CreateCollisionShapeAddMesh(command, file_ptr, scale.as_ptr()) }
            }
            ConvexMesh { vertices, scale } => {
                if vertices.is_empty() {
                    return Err(BulletError::CommandFailed {
                        message: "Convex mesh requires at least one vertex",
                        code: -1,
                    });
                }
                let flattened = Self::flatten_points(vertices);
                let vertices_ptr = scratch.push_f64_buffer(flattened);
                unsafe {
                    ffi::b3CreateCollisionShapeAddConvexMesh(
                        self.handle,
                        command,
                        scale.as_ptr(),
                        vertices_ptr,
                        vertices.len() as i32,
                    )
                }
            }
            Mesh { vertices, indices, scale } => {
                if vertices.is_empty() || indices.is_empty() {
                    return Err(BulletError::CommandFailed {
                        message: "Concave mesh requires vertices and indices",
                        code: -1,
                    });
                }
                let vertex_buffer = Self::flatten_points(vertices);
                let index_buffer = indices.to_vec();
                let vertices_ptr = scratch.push_f64_buffer(vertex_buffer);
                let indices_ptr = scratch.push_i32_buffer(index_buffer);
                unsafe {
                    ffi::b3CreateCollisionShapeAddConcaveMesh(
                        self.handle,
                        command,
                        scale.as_ptr(),
                        vertices_ptr,
                        vertices.len() as i32,
                        indices_ptr,
                        indices.len() as i32,
                    )
                }
            }
            HeightfieldFile { file, mesh_scale, texture_scaling } => {
                let file_ptr = scratch.push_c_string(file)?;
                unsafe {
                    ffi::b3CreateCollisionShapeAddHeightfield(
                        command,
                        file_ptr,
                        mesh_scale.as_ptr(),
                        *texture_scaling,
                    )
                }
            }
            HeightfieldData {
                mesh_scale,
                texture_scaling,
                samples,
                rows,
                columns,
                replace_index,
            } => {
                if *rows <= 0 || *columns <= 0 {
                    return Err(BulletError::CommandFailed {
                        message: "Heightfield dimensions must be positive",
                        code: -1,
                    });
                }
                let expected_len = (*rows as usize).saturating_mul(*columns as usize);
                if expected_len != samples.len() {
                    return Err(BulletError::CommandFailed {
                        message: "Heightfield sample count mismatch",
                        code: expected_len as i32,
                    });
                }
                let buffer = samples.to_vec();
                let data_ptr = scratch.push_f32_buffer_mut(buffer);
                unsafe {
                    ffi::b3CreateCollisionShapeAddHeightfield2(
                        self.handle,
                        command,
                        mesh_scale.as_ptr(),
                        *texture_scaling,
                        data_ptr,
                        *rows,
                        *columns,
                        replace_index.unwrap_or(-1),
                    )
                }
            }
        };

        if index < 0 {
            Err(BulletError::CommandFailed {
                message: "Bullet rejected collision geometry",
                code: index,
            })
        } else {
            Ok(index)
        }
    }

    fn add_visual_geometry(
        &self,
        command: ffi::b3SharedMemoryCommandHandle,
        geometry: &VisualGeometry<'_>,
        scratch: &mut GeometryScratch,
    ) -> BulletResult<i32> {
        use VisualGeometry::*;

        let index = match geometry {
            Sphere { radius } => {
                if *radius <= 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Sphere radius must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateVisualShapeAddSphere(command, *radius) }
            }
            Box { half_extents } => unsafe {
                ffi::b3CreateVisualShapeAddBox(command, half_extents.as_ptr())
            },
            Capsule { radius, height } => {
                if *radius <= 0.0 || *height < 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Capsule dimensions must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateVisualShapeAddCapsule(command, *radius, *height) }
            }
            Cylinder { radius, height } => {
                if *radius <= 0.0 || *height <= 0.0 {
                    return Err(BulletError::CommandFailed {
                        message: "Cylinder dimensions must be positive",
                        code: -1,
                    });
                }
                unsafe { ffi::b3CreateVisualShapeAddCylinder(command, *radius, *height) }
            }
            Plane { normal, constant } => unsafe {
                ffi::b3CreateVisualShapeAddPlane(command, normal.as_ptr(), *constant)
            },
            Mesh { file, scale } => {
                let file_ptr = scratch.push_c_string(file)?;
                unsafe { ffi::b3CreateVisualShapeAddMesh(command, file_ptr, scale.as_ptr()) }
            }
            MeshData { vertices, indices, normals, uvs, scale } => {
                if vertices.is_empty() {
                    return Err(BulletError::CommandFailed {
                        message: "Visual mesh requires vertices",
                        code: -1,
                    });
                }
                let vertex_buffer = Self::flatten_points(vertices);
                let vertices_ptr = scratch.push_f64_buffer(vertex_buffer);

                let (indices_ptr, num_indices) = if let Some(indices) = indices {
                    let buffer = indices.to_vec();
                    (scratch.push_i32_buffer(buffer), indices.len() as i32)
                } else {
                    (ptr::null::<i32>(), 0)
                };

                let (normals_ptr, num_normals) = if let Some(normals) = normals {
                    let buffer = Self::flatten_points(normals);
                    (scratch.push_f64_buffer(buffer), normals.len() as i32)
                } else {
                    (ptr::null::<f64>(), 0)
                };

                let (uvs_ptr, num_uvs) = if let Some(uvs) = uvs {
                    let buffer = Self::flatten_points(uvs);
                    (scratch.push_f64_buffer(buffer), (uvs.len() * 2) as i32)
                } else {
                    (ptr::null::<f64>(), 0)
                };

                unsafe {
                    ffi::b3CreateVisualShapeAddMesh2(
                        self.handle,
                        command,
                        scale.as_ptr(),
                        vertices_ptr,
                        vertices.len() as i32,
                        indices_ptr,
                        num_indices,
                        normals_ptr,
                        num_normals,
                        uvs_ptr,
                        num_uvs,
                    )
                }
            }
        };

        if index < 0 {
            Err(BulletError::CommandFailed {
                message: "Bullet rejected visual geometry",
                code: index,
            })
        } else {
            Ok(index)
        }
    }

    fn flatten_points<const N: usize>(values: &[[f64; N]]) -> Vec<f64> {
        let mut out = Vec::with_capacity(values.len() * N);
        for value in values {
            out.extend_from_slice(value);
        }
        out
    }

    fn flatten_isometries(values: &[na::Isometry3<f64>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(values.len() * 7);
        for transform in values {
            let (position, orientation) = isometry_to_raw_parts(transform);
            out.extend_from_slice(&position);
            out.extend_from_slice(&orientation);
        }
        out
    }

    fn write_transform_to_frame(transform: &na::Isometry3<f64>, frame: &mut [f64; 7]) {
        isometry_write_to_frame(transform, frame);
    }

    fn read_frame_transform(frame: &[f64; 7]) -> na::Isometry3<f64> {
        isometry_from_frame(frame)
    }

    fn constraint_update_has_changes(update: &ConstraintUpdate) -> bool {
        update.child_frame.is_some()
            || update.max_force.is_some()
            || update.gear_ratio.is_some()
            || update.gear_aux_link.is_some()
            || update.relative_position_target.is_some()
            || update.erp.is_some()
    }

    pub(crate) fn read_c_string(raw: &[std::os::raw::c_char]) -> String {
        if raw.is_empty() || raw[0] == 0 {
            return String::new();
        }
        unsafe { CStr::from_ptr(raw.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn request_actual_state_status(&mut self, body_unique_id: i32) -> BulletResult<CommandStatus> {
        self.request_actual_state_status_with_flags(body_unique_id, false, false)
    }

    fn request_actual_state_status_with_flags(
        &mut self,
        body_unique_id: i32,
        compute_link_velocity: bool,
        compute_forward_kinematics: bool,
    ) -> BulletResult<CommandStatus> {
        self.ensure_can_submit()?;
        let command = unsafe { ffi::b3RequestActualStateCommandInit(self.handle, body_unique_id) };
        unsafe {
            let velocity_flag = compute_link_velocity.into();
            let kinematics_flag = compute_forward_kinematics.into();
            ffi::b3RequestActualStateCommandComputeLinkVelocity(command, velocity_flag);
            ffi::b3RequestActualStateCommandComputeForwardKinematics(command, kinematics_flag);
        }
        self.submit_simple_command(
            command,
            ffi::EnumSharedMemoryServerStatus::CMD_ACTUAL_STATE_UPDATE_COMPLETED,
        )
    }

    fn read_joint_state(
        &mut self,
        status_handle: ffi::b3SharedMemoryStatusHandle,
        joint_index: i32,
    ) -> BulletResult<JointState> {
        let mut raw = MaybeUninit::<ffi::b3JointSensorState>::uninit();
        let success = unsafe {
            ffi::b3GetJointState(self.handle, status_handle, joint_index, raw.as_mut_ptr())
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Cannot query joint state",
                code: success,
            });
        }
        let raw = unsafe { raw.assume_init() };
        Ok(JointState {
            position: raw.m_joint_position,
            velocity: raw.m_joint_velocity,
            force_torque: raw.m_joint_force_torque,
            motor_torque: raw.m_joint_motor_torque,
        })
    }

    fn read_joint_state_multi_dof(
        &mut self,
        status_handle: ffi::b3SharedMemoryStatusHandle,
        joint_index: i32,
    ) -> BulletResult<JointStateMultiDof> {
        let mut raw = MaybeUninit::<ffi::b3JointSensorState2>::uninit();
        let success = unsafe {
            ffi::b3GetJointStateMultiDof(self.handle, status_handle, joint_index, raw.as_mut_ptr())
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Cannot query multi-DoF joint state",
                code: success,
            });
        }
        let raw = unsafe { raw.assume_init() };

        let q_len = raw.m_q_dof_size as usize;
        let u_len = raw.m_u_dof_size as usize;
        if q_len > raw.m_joint_position.len() || u_len > raw.m_joint_velocity.len() {
            return Err(BulletError::CommandFailed {
                message: "Bullet reported invalid multi-DoF state lengths",
                code: (q_len.max(u_len)) as i32,
            });
        }

        Ok(JointStateMultiDof {
            positions: raw.m_joint_position[..q_len].to_vec(),
            velocities: raw.m_joint_velocity[..u_len].to_vec(),
            reaction_forces: raw.m_joint_reaction_force_torque,
            motor_torques: raw.m_joint_motor_torque_multi_dof[..u_len].to_vec(),
        })
    }

    fn read_link_state(
        &mut self,
        status_handle: ffi::b3SharedMemoryStatusHandle,
        link_index: i32,
    ) -> BulletResult<LinkState> {
        let mut raw = MaybeUninit::<ffi::b3LinkState>::uninit();
        let success = unsafe {
            ffi::b3GetLinkState(self.handle, status_handle, link_index, raw.as_mut_ptr())
        };
        if success == 0 {
            return Err(BulletError::CommandFailed {
                message: "Cannot query link state",
                code: success,
            });
        }
        unsafe { raw.assume_init() }.try_into()
    }

    fn extract_base_state(
        status_handle: ffi::b3SharedMemoryStatusHandle,
    ) -> BulletResult<(na::Isometry3<f64>, [f64; 6])> {
        if status_handle.is_null() {
            return Err(BulletError::NullPointer(
                "Bullet returned a null status handle while reading base state",
            ));
        }

        let mut actual_state_q: *const f64 = ptr::null();
        let mut actual_state_qdot: *const f64 = ptr::null();

        let result = unsafe {
            ffi::b3GetStatusActualState(
                status_handle,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut actual_state_q,
                &raw mut actual_state_qdot,
                ptr::null_mut(),
            )
        };

        if result == 0 {
            return Err(BulletError::CommandFailed {
                message: "Failed to read actual state",
                code: result,
            });
        }

        if actual_state_q.is_null() {
            return Err(BulletError::NullPointer(
                "Bullet returned a null actual-state position pointer",
            ));
        }

        let mut position = [0.0; 3];
        let mut orientation = [0.0; 4];
        unsafe {
            ptr::copy_nonoverlapping(actual_state_q, position.as_mut_ptr(), 3);
            ptr::copy_nonoverlapping(actual_state_q.add(3), orientation.as_mut_ptr(), 4);
        }

        let mut velocity = [0.0; 6];
        if !actual_state_qdot.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(actual_state_qdot, velocity.as_mut_ptr(), 6);
            }
        }

        Ok((
            na::Isometry3::from_parts(
                na::Translation3::from(position),
                na::UnitQuaternion::from_quaternion(orientation.into()),
            ),
            velocity,
        ))
    }
}

impl Drop for PhysicsClient {
    fn drop(&mut self) {
        unsafe {
            ffi::b3DisconnectSharedMemory(self.handle);
        }
        // Decrement totals
        let _ = TOTAL_CLIENTS.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            Some(v.saturating_sub(1))
        });
        if let Mode::Gui | Mode::GuiServer = self.mode {
            // Release GUI slot
            GUI_GUARD.store(0, Ordering::SeqCst);
            let _ = GUI_CLIENTS.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                Some(v.saturating_sub(1))
            });
        }
    }
}

// ---- Process-wide statistics ----
/// Return the number of active physics clients in this process (PyBullet-compatible name).
pub fn get_num_physics_clients() -> usize {
    TOTAL_CLIENTS.load(Ordering::SeqCst)
}

/// Return the number of active GUI/graphics clients in this process (PyBullet-compatible name).
pub fn get_num_gui_physics_clients() -> usize {
    GUI_CLIENTS.load(Ordering::SeqCst)
}
