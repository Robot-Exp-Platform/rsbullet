use std::time::Duration;

use nalgebra as na;
use robot_behavior::Entity;
use rsbullet_sys::{self as ffi, b3JointInfo, b3KeyboardEvent, b3LinkState, b3MouseEvent};

use crate::{BulletError, PhysicsClient};

#[derive(Debug)]
pub(crate) struct CommandStatus {
    pub(crate) handle: ffi::b3SharedMemoryStatusHandle,
    pub(crate) status_type: i32,
}

// ### Connection & Session
//
// | API | Status | Notes |
// | --- | --- | --- |
// | connect | **Implemented** | `PhysicsClient::connect` covers all modes in scope |
// | disconnect | **Implemented** | `PhysicsClient::disconnectPhysicsServer` + `Drop` |
// | getConnectionInfo | **Implemented** | Requires client/server introspection |
// | isConnected | **Implemented** | Uses `b3CanSubmitCommand` |
//

// ### Simulation Parameters
//
// | API | Status | Notes |
// | --- | --- | --- |
// | resetSimulation | **Implemented** | Core reset command |
// | stepSimulation | **Implemented** | Core stepping command |
// | performCollisionDetection | **Implemented** | Extra collision command |
// | setGravity | **Implemented** | Physics param command |
// | setTimeStep | **Implemented** | Physics param command |
// | setDefaultContactERP | **Implemented** | Physics param command |
// | setRealTimeSimulation | **Implemented** | Physics param command |
// | setPhysicsEngineParameter | **Implemented** | Bulk physics-param configuration |
// | getPhysicsEngineParameters | **Implemented** | Query mirror of above |
// | setInternalSimFlags | **Implemented** | Expert-only flagging |

/// Options for the [`set_physics_engine_parameter`](`crate::PhysicsClient::set_physics_engine_parameter`) method.
#[derive(Default, Debug)]
pub struct PhysicsEngineParametersUpdate {
    /// See the warning in the [`set_time_step`](`crate::PhysicsClient::set_time_step`) section.
    /// physics engine time step,
    /// each time you call [`step_simulation`](`crate::PhysicsClient::step_simulation`) simulated
    /// time will progress this amount. Same as [`set_time_step`](`crate::PhysicsClient::set_time_step`)
    pub fixed_time_step: Option<Duration>,
    ///Choose the maximum number of constraint solver iterations.
    /// If the `solver_residual_threshold` is reached,
    /// the solver may terminate before the `num_solver_iterations`.
    pub num_solver_iterations: Option<usize>,
    /// Advanced feature, only when using maximal coordinates: split the positional
    /// constraint solving and velocity constraint solving in two stages,
    /// to prevent huge penetration recovery forces.
    pub use_split_impulse: Option<bool>,
    /// Related to `use_split_impulse`: if the penetration for a particular contact constraint is
    /// less than this specified threshold, no split impulse will happen for that contact.
    pub split_impulse_penetration_threshold: Option<f64>,
    /// Subdivide the physics simulation step further by `num_sub_steps`.
    /// This will trade performance over accuracy.
    pub num_sub_steps: Option<usize>,
    /// Use 0 for default collision filter: (group A&maskB) AND (groupB&maskA).
    /// Use 1 to switch to the OR collision filter: (group A&maskB) OR (groupB&maskA)
    pub collision_filter_mode: Option<usize>,
    /// Contact points with distance exceeding this threshold are not processed by the LCP solver.
    /// In addition, AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x.
    pub contact_breaking_threshold: Option<f64>,
    /// Experimental: add 1ms sleep if the number of commands executed exceed this threshold.
    /// setting the value to `-1` disables the feature.
    pub max_num_cmd_per_1_ms: Option<i32>,
    /// Set to `false` to disable file caching, such as .obj wavefront file loading
    pub enable_file_caching: Option<bool>,
    /// If relative velocity is below this threshold, restitution will be zero.
    pub restitution_velocity_threshold: Option<f64>,
    /// constraint error reduction parameter (non-contact, non-friction)
    pub erp: Option<f64>,
    /// contact error reduction parameter
    pub contact_erp: Option<f64>,
    /// friction error reduction parameter (when positional friction anchors are enabled)
    pub friction_erp: Option<f64>,
    /// Set to `false` to disable implicit cone friction and use pyramid approximation (cone is default).
    /// NOTE: Although enabled by default, it is worth trying to disable this feature, in case there are friction artifacts.
    pub enable_cone_friction: Option<bool>,
    /// enables or disables sorting of overlapping pairs (backward compatibility setting).
    pub deterministic_overlapping_pairs: Option<bool>,
    /// If continuous collision detection (CCD) is enabled, CCD will not be used if the
    /// penetration is below this threshold.
    pub allowed_ccd_penetration: Option<f64>,
    /// Specifcy joint feedback frame
    pub joint_feedback_mode: Option<JointFeedbackMode>,
    /// velocity threshold, if the maximum velocity-level error for each constraint is below this
    /// threshold the solver will terminate (unless the solver hits the numSolverIterations).
    /// Default value is 1e-7
    pub solver_residual_threshold: Option<f64>,
    /// Position correction of contacts is not resolved below this threshold,
    /// to allow more stable contact.
    pub contact_slop: Option<f64>,
    /// if true, enable separating axis theorem based convex collision detection,
    /// if features are available (instead of using GJK and EPA).
    /// Requires [`URDF_INITIALIZE_SAT_FEATURES`](`LoadModelFlags::URDF_INITIALIZE_SAT_FEATURES`) in
    /// the [`UrdfOptions`](`UrdfOptions`) in [`load_urdf`](`crate::PhysicsClient::load_urdf`).
    pub enable_sat: Option<bool>,
    /// Experimental (best to ignore): allow to use a direct LCP solver, such as Dantzig.
    pub constraint_solver_type: Option<ConstraintSolverType>,
    /// Experimental (best to ignore) global default constraint force mixing parameter.
    pub global_cfm: Option<f64>,
    /// Experimental (best to ignore), minimum size of constraint solving islands,
    /// to avoid very small islands of independent constraints.
    pub minimum_solver_island_size: Option<usize>,
    /// when true, additional solve analytics is available.
    pub report_solver_analytics: Option<bool>,
    /// fraction of previous-frame force/impulse that is used to initialize the initial solver solution
    pub warm_starting_factor: Option<f64>,
    pub sparse_sdf_voxel_size: Option<f64>,
    pub num_non_contact_inner_iterations: Option<usize>,
}
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub enum ConstraintSolverType {
    None,
    Si = 1,
    Pgs,
    Dantzig,
    Lemke,
    Nncg,
    BlockPgs,
}

impl TryFrom<i32> for ConstraintSolverType {
    type Error = BulletError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ConstraintSolverType::None),
            1 => Ok(ConstraintSolverType::Si),
            2 => Ok(ConstraintSolverType::Pgs),
            3 => Ok(ConstraintSolverType::Dantzig),
            4 => Ok(ConstraintSolverType::Lemke),
            5 => Ok(ConstraintSolverType::Nncg),
            6 => Ok(ConstraintSolverType::BlockPgs),
            _ => Err(BulletError::UnknownType("Unexpected ConstraintSolverType")),
        }
    }
}

/// Specifies joint feedback frame. Is used in
/// [`SetPhysicsEngineParameterOptions::joint_feedback_mode`](`SetPhysicsEngineParameterOptions::joint_feedback_mode`)
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub enum JointFeedbackMode {
    None,
    /// gets the joint feedback in world space
    WorldSpace = 1,
    /// gets the joint feedback in the joint frame
    JointFrame,
}

impl TryFrom<i32> for JointFeedbackMode {
    type Error = BulletError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(JointFeedbackMode::None),
            1 => Ok(JointFeedbackMode::WorldSpace),
            2 => Ok(JointFeedbackMode::JointFrame),
            _ => Err(BulletError::UnknownType("Unexpected JointFeedbackMode")),
        }
    }
}

/// See [`SetPhysicsEngineParameterOptions`](`SetPhysicsEngineParameterOptions`) for a description of the parameters.
#[derive(Debug, Clone)]
pub struct PhysicsEngineParameters {
    pub fixed_time_step: Duration,
    pub simulation_time_stamp: Duration,
    pub num_solver_iterations: usize,
    pub use_split_impulse: bool,
    pub split_impulse_penetration_threshold: f64,
    pub num_sub_steps: usize,
    pub collision_filter_mode: usize,
    pub contact_breaking_threshold: f64,
    pub enable_file_caching: bool,
    pub restitution_velocity_threshold: f64,
    pub erp: f64,
    pub contact_erp: f64,
    pub friction_erp: f64,
    pub enable_cone_friction: bool,
    pub deterministic_overlapping_pairs: bool,
    pub allowed_ccd_penetration: f64,
    pub joint_feedback_mode: JointFeedbackMode,
    pub solver_residual_threshold: f64,
    pub contact_slop: f64,
    pub enable_sat: bool,
    pub constraint_solver_type: ConstraintSolverType,
    pub global_cfm: f64,
    pub minimum_solver_island_size: usize,
    pub report_solver_analytics: bool,
    pub warm_starting_factor: f64,
    pub sparse_sdf_voxel_size: f64,
    pub num_non_contact_inner_iterations: usize,
    pub use_real_time_simulation: bool,
    pub gravity: [f64; 3],
    pub articulated_warm_starting_factor: f64,
    pub internal_sim_flags: i32,
    pub friction_cfm: f64,
}

impl TryFrom<ffi::b3PhysicsSimulationParameters> for PhysicsEngineParameters {
    type Error = BulletError;

    fn try_from(b3: ffi::b3PhysicsSimulationParameters) -> Result<Self, Self::Error> {
        Ok(PhysicsEngineParameters {
            fixed_time_step: Duration::from_secs_f64(b3.m_deltaTime),
            simulation_time_stamp: Duration::from_secs_f64(b3.m_simulationTimestamp),
            num_solver_iterations: b3.m_numSolverIterations as usize,
            use_split_impulse: b3.m_useSplitImpulse != 0,
            split_impulse_penetration_threshold: b3.m_splitImpulsePenetrationThreshold,
            num_sub_steps: b3.m_numSimulationSubSteps as usize,
            collision_filter_mode: b3.m_collisionFilterMode as usize,
            contact_breaking_threshold: b3.m_contactBreakingThreshold,
            enable_file_caching: b3.m_enableFileCaching != 0,
            restitution_velocity_threshold: b3.m_restitutionVelocityThreshold,
            erp: b3.m_defaultNonContactERP,
            contact_erp: b3.m_defaultContactERP,
            friction_erp: b3.m_frictionERP,
            enable_cone_friction: b3.m_enableConeFriction != 0,
            deterministic_overlapping_pairs: b3.m_deterministicOverlappingPairs != 0,
            allowed_ccd_penetration: b3.m_allowedCcdPenetration,
            joint_feedback_mode: b3.m_jointFeedbackMode.try_into()?,
            solver_residual_threshold: b3.m_solverResidualThreshold,
            contact_slop: b3.m_contactSlop,
            enable_sat: b3.m_enableSAT != 0,
            constraint_solver_type: b3.m_constraintSolverType.try_into()?,
            global_cfm: b3.m_defaultGlobalCFM,
            minimum_solver_island_size: b3.m_minimumSolverIslandSize as usize,
            report_solver_analytics: b3.m_reportSolverAnalytics != 0,
            warm_starting_factor: b3.m_warmStartingFactor,
            sparse_sdf_voxel_size: b3.m_sparseSdfVoxelSize,
            num_non_contact_inner_iterations: b3.m_numNonContactInnerIterations as usize,
            use_real_time_simulation: b3.m_useRealTimeSimulation != 0,
            gravity: b3.m_gravityAcceleration,
            articulated_warm_starting_factor: b3.m_articulatedWarmStartingFactor,
            internal_sim_flags: b3.m_internalSimFlags,
            friction_cfm: b3.m_frictionCFM,
        })
    }
}

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

#[derive(Debug, Clone, Default)]
pub struct UrdfOptions {
    pub base: Option<na::Isometry3<f64>>,
    pub use_maximal_coordinates: Option<bool>,
    pub use_fixed_base: bool,
    pub flags: Option<LoadModelFlags>,
    pub global_scaling: Option<f64>,
}

impl From<()> for UrdfOptions {
    fn from((): ()) -> Self {
        Self::default()
    }
}

impl From<[f64; 3]> for UrdfOptions {
    fn from(base: [f64; 3]) -> Self {
        Self {
            base: Some(na::Isometry3::translation(base[0], base[1], base[2])),
            ..Self::default()
        }
    }
}

impl From<na::Isometry3<f64>> for UrdfOptions {
    fn from(base: na::Isometry3<f64>) -> Self {
        Self { base: Some(base), ..Self::default() }
    }
}

impl From<LoadModelFlags> for UrdfOptions {
    fn from(flags: LoadModelFlags) -> Self {
        UrdfOptions { flags: Some(flags), ..Default::default() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SdfOptions {
    pub use_maximal_coordinates: Option<bool>,
    pub global_scaling: Option<f64>,
}

impl From<(Option<bool>, Option<f64>)> for SdfOptions {
    fn from(value: (Option<bool>, Option<f64>)) -> Self {
        Self { use_maximal_coordinates: value.0, global_scaling: value.1 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MjcfOptions {
    pub flags: Option<LoadModelFlags>,
    pub use_multi_body: Option<bool>,
}

impl From<LoadModelFlags> for Option<MjcfOptions> {
    fn from(flags: LoadModelFlags) -> Self {
        Some(MjcfOptions { flags: Some(flags), use_multi_body: None })
    }
}

bitflags::bitflags! {
    /// Use flag for loading the model. Flags can be combined with the `|`-operator.
    /// Example:
    /// ```rust
    ///# use rsbullet_core::LoadModelFlags;
    /// let flags = LoadModelFlags::URDF_ENABLE_CACHED_GRAPHICS_SHAPES | LoadModelFlags::URDF_PRINT_URDF_INFO;
    /// assert!(flags.contains(LoadModelFlags::URDF_PRINT_URDF_INFO));
    /// ```
    #[derive(Debug, Clone)]
    pub struct LoadModelFlags : i32 {
        /// Use the inertia tensor provided in the URDF.
        ///
        /// By default, Bullet will recompute the inertial tensor based on the mass and volume of the
        /// collision shape. Use this is you can provide a more accurate inertia tensor.
        const URDF_USE_INERTIA_FROM_FILE = 1 << 1;
        /// Enables self-collision.
        const URDF_USE_SELF_COLLISION = 1 << 3;
        const URDF_USE_SELF_COLLISION_EXCLUDE_PARENT = 1 << 4;
        /// will discard self-collisions between a child link and any of its ancestors
        /// (parents, parents of parents, up to the base).
        /// Needs to be used together with [`URDF_USE_SELF_COLLISION`](`Self::URDF_USE_SELF_COLLISION`).
        const URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS = 1 << 5;
        const URDF_RESERVED = 1 << 6;
        /// will use a smooth implicit cylinder. By default, Bullet will tesselate the cylinder
        /// into a convex hull.
        const URDF_USE_IMPLICIT_CYLINDER = 1 << 7;
        const URDF_GLOBAL_VELOCITIES_MB = 1 << 8;
        const MJCF_COLORS_FROM_FILE = 1 << 9;
        /// Caches as reuses graphics shapes. This will decrease loading times for similar objects
        const URDF_ENABLE_CACHED_GRAPHICS_SHAPES = 1 << 10;
        /// Allow the disabling of simulation after a body hasn't moved for a while.
        ///
        /// Interaction with active bodies will re-enable simulation.
        const URDF_ENABLE_SLEEPING = 1 << 11;
        /// will create triangle meshes for convex shapes. This will improve visualization and also
        /// allow usage of the separating axis test (SAT) instead of GJK/EPA.
        /// Requires to enable_SAT using set_physics_engine_parameter. TODO
        const URDF_INITIALIZE_SAT_FEATURES = 1 << 12;
        /// will enable collision between child and parent, it is disabled by default.
        /// Needs to be used together with [`URDF_USE_SELF_COLLISION`](`Self::URDF_USE_SELF_COLLISION`) flag.
        const URDF_USE_SELF_COLLISION_INCLUDE_PARENT = 1 << 13;
        const URDF_PARSE_SENSORS = 1 << 14;
        /// will use the RGB color from the Wavefront OBJ file, instead of from the URDF file.
        const URDF_USE_MATERIAL_COLORS_FROM_MTL = 1 << 15;
        const URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL = 1 << 16;
        /// Try to maintain the link order from the URDF file.
        const URDF_MAINTAIN_LINK_ORDER = 1 << 17;
        const URDF_ENABLE_WAKEUP = 1 << 18;
        /// this will remove fixed links from the URDF file and merge the resulting links.
        /// This is good for performance, since various algorithms
        /// (articulated body algorithm, forward kinematics etc) have linear complexity
        /// in the number of joints, including fixed joints.
        const URDF_MERGE_FIXED_LINKS = 1 << 19;
        const URDF_IGNORE_VISUAL_SHAPES = 1 << 20;
        const URDF_IGNORE_COLLISION_SHAPES = 1 << 21;
        const URDF_PRINT_URDF_INFO = 1 << 22;
        const URDF_GOOGLEY_UNDEFINED_COLORS = 1 << 23;
    }
}

// ### Asset Creation & Mutation
//
// | API | Status | Notes |
// | --- | --- | --- |
// | createCollisionShape | **Implemented** | Supports primitive/mesh geometry |
// | createCollisionShapeArray | **Implemented** | Compound shape builder |
// | removeCollisionShape | **Implemented** | Clean-up helper |
// | getMeshData | Pending | Mesh inspection |
// | getTetraMeshData | Pending | Soft-body mesh |
// | resetMeshData | Optional | Deformable specific |
// | createVisualShape | **Implemented** | Visual geometry authoring |
// | createVisualShapeArray | **Implemented** | Bulk visual authoring |
// | createMultiBody | **Implemented** | Procedural multibody build |
// | createConstraint | **Implemented** | Constraint authoring |
// | changeConstraint | **Implemented** | Constraint mutation |
// | removeConstraint | **Implemented** | Constraint teardown |
// | enableJointForceTorqueSensor | **Implemented** | Sensor toggle |
// | removeBody | **Implemented** | Body teardown |
// | getNumConstraints | **Implemented** | Constraint enumeration |
// | getConstraintInfo | **Implemented** | Constraint query |
// | getConstraintState | **Implemented** | Constraint forces |
// | getConstraintUniqueId | **Implemented** | Constraint enumeration |
// | changeVisualShape | **Implemented** | Visual mutation |
// | resetVisualShapeData | Pending | Legacy alias |
// | loadTexture | **Implemented** | Visual assets |
// | changeTexture | **Implemented** | Visual assets |

/// The unique ID for a Collision Shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CollisionId(pub i32);

impl Default for CollisionId {
    fn default() -> Self {
        Self(-1)
    }
}

#[derive(Debug, Clone)]
pub enum CollisionGeometry<'a> {
    Sphere {
        radius: f64,
    },
    Box {
        half_extents: [f64; 3],
    },
    Capsule {
        radius: f64,
        height: f64,
    },
    Cylinder {
        radius: f64,
        height: f64,
    },
    Plane {
        normal: [f64; 3],
        constant: f64,
    },
    MeshFile {
        file: &'a str,
        scale: [f64; 3],
    },
    ConvexMesh {
        vertices: &'a [[f64; 3]],
        scale: [f64; 3],
    },
    Mesh {
        vertices: &'a [[f64; 3]],
        indices: &'a [i32],
        scale: [f64; 3],
    },
    HeightfieldFile {
        file: &'a str,
        mesh_scale: [f64; 3],
        texture_scaling: f64,
    },
    HeightfieldData {
        mesh_scale: [f64; 3],
        texture_scaling: f64,
        samples: &'a [f32],
        rows: i32,
        columns: i32,
        replace_index: Option<i32>,
    },
}

impl<'a> From<Entity<'a>> for CollisionGeometry<'a> {
    fn from(entity: Entity<'a>) -> Self {
        match entity {
            Entity::Sphere { radius } => Self::Sphere { radius },
            Entity::Box { half_extents } => Self::Box { half_extents },
            Entity::Capsule { radius, height } => Self::Capsule { radius, height },
            Entity::Cylinder { radius, height } => Self::Cylinder { radius, height },
            Entity::Plane { normal, constant } => Self::Plane { normal, constant },
            Entity::MeshFile { file, scale } => Self::MeshFile { file, scale },
            Entity::ConvexMesh { vertices, scale } => Self::ConvexMesh { vertices, scale },
            Entity::Mesh { vertices, indices, scale } => Self::Mesh { vertices, indices, scale },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CollisionShapeOptions {
    pub transform: na::Isometry3<f64>,
    pub flags: Option<i32>,
}

impl Default for CollisionShapeOptions {
    fn default() -> Self {
        Self { transform: na::Isometry3::identity(), flags: None }
    }
}

impl From<na::Isometry3<f64>> for CollisionShapeOptions {
    fn from(transform: na::Isometry3<f64>) -> Self {
        Self { transform, flags: None }
    }
}

impl<T> From<(T, Option<i32>)> for CollisionShapeOptions
where
    T: Into<na::Isometry3<f64>>,
{
    fn from(value: (T, Option<i32>)) -> Self {
        Self { transform: value.0.into(), flags: value.1 }
    }
}

/// The unique ID for a Visual Shape
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VisualId(pub i32);

impl Default for VisualId {
    fn default() -> Self {
        Self(-1)
    }
}

#[derive(Debug, Clone)]
pub enum VisualGeometry<'a> {
    Sphere {
        radius: f64,
    },
    Box {
        half_extents: [f64; 3],
    },
    Capsule {
        radius: f64,
        height: f64,
    },
    Cylinder {
        radius: f64,
        height: f64,
    },
    Plane {
        normal: [f64; 3],
        constant: f64,
    },
    Mesh {
        file: &'a str,
        scale: [f64; 3],
    },
    MeshData {
        vertices: &'a [[f64; 3]],
        indices: Option<&'a [i32]>,
        normals: Option<&'a [[f64; 3]]>,
        uvs: Option<&'a [[f64; 2]]>,
        scale: [f64; 3],
    },
}

impl<'a> From<Entity<'a>> for VisualGeometry<'a> {
    fn from(entity: Entity<'a>) -> Self {
        match entity {
            Entity::Sphere { radius } => Self::Sphere { radius },
            Entity::Box { half_extents } => Self::Box { half_extents },
            Entity::Capsule { radius, height } => Self::Capsule { radius, height },
            Entity::Cylinder { radius, height } => Self::Cylinder { radius, height },
            Entity::Plane { normal, constant } => Self::Plane { normal, constant },
            Entity::MeshFile { file, scale } => Self::Mesh { file, scale },
            Entity::Mesh { vertices, indices, scale } => Self::MeshData {
                vertices,
                indices: Some(indices),
                normals: None,
                uvs: None,
                scale,
            },
            Entity::ConvexMesh { vertices, scale } => {
                Self::MeshData { vertices, indices: None, normals: None, uvs: None, scale }
            }
        }
    }
}

bitflags::bitflags! {
    /// Experimental flags, best to ignore.
    #[derive(Debug, Clone, Copy)]
    pub struct VisualShapeFlags : i32 {
        const TEXTURE_UNIQUE_IDS = 1;
        const DOUBLE_SIDED = 4;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VisualShapeOptions {
    pub rgba: [f64; 4],
    pub specular: [f64; 3],
    pub flags: Option<VisualShapeFlags>,
    pub transform: na::Isometry3<f64>,
}

impl Default for VisualShapeOptions {
    fn default() -> Self {
        Self {
            rgba: [1.0, 1.0, 1.0, 1.0],
            specular: [1.0, 1.0, 1.0],
            flags: None,
            transform: na::Isometry3::identity(),
        }
    }
}

impl From<na::Isometry3<f64>> for VisualShapeOptions {
    fn from(transform: na::Isometry3<f64>) -> Self {
        Self { transform, ..Default::default() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChangeVisualShapeOptions {
    pub shape_index: VisualId,
    pub texture_unique_id: Option<i32>,
    pub rgba_color: Option<[f64; 4]>,
    pub specular_color: Option<[f64; 3]>,
    pub flags: Option<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct MultiBodyBase {
    pub mass: f64,
    pub pose: na::Isometry3<f64>,
    pub collision_shape: CollisionId,
    pub visual_shape: VisualId,
    pub inertial_pose: na::Isometry3<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct MultiBodyLink {
    pub mass: f64,
    pub collision_shape: CollisionId,
    pub visual_shape: VisualId,
    pub parent_index: Option<usize>,
    pub joint_type: JointType,
    pub joint_axis: [f64; 3],
    pub parent_transform: na::Isometry3<f64>,
    pub inertial_transform: na::Isometry3<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct MultiBodyCreateOptions<'a> {
    pub base: MultiBodyBase,
    pub links: &'a [MultiBodyLink],
    pub use_maximal_coordinates: bool,
    pub flags: Option<i32>,
    pub batch_transforms: Option<&'a [na::Isometry3<f64>]>,
}

#[derive(Debug, Clone, Default)]
pub struct ConstraintCreateOptions {
    pub parent_body: i32,
    pub parent_link: Option<usize>,
    pub child_body: i32,
    pub child_link: Option<usize>,
    pub joint_type: JointType,
    pub parent_frame: na::Isometry3<f64>,
    pub child_frame: na::Isometry3<f64>,
    pub joint_axis: [f64; 3],
    pub max_applied_force: f64,
    pub gear_ratio: Option<f64>,
    pub gear_aux_link: Option<usize>,
    pub relative_position_target: Option<f64>,
    pub erp: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ConstraintUpdate {
    pub child_frame: Option<na::Isometry3<f64>>,
    pub max_force: Option<f64>,
    pub gear_ratio: Option<f64>,
    pub gear_aux_link: Option<usize>,
    pub relative_position_target: Option<f64>,
    pub erp: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    pub parent_body: i32,
    pub parent_link: i32,
    pub child_body: i32,
    pub child_link: i32,
    pub parent_frame: na::Isometry3<f64>,
    pub child_frame: na::Isometry3<f64>,
    pub joint_axis: [f64; 3],
    pub joint_type: i32,
    pub max_applied_force: f64,
    pub constraint_unique_id: i32,
    pub gear_ratio: f64,
    pub gear_aux_link: i32,
    pub relative_position_target: f64,
    pub erp: f64,
}

#[derive(Debug, Clone)]
pub struct ConstraintState {
    pub applied_forces: [f64; 6],
    pub dof_count: i32,
}

#[derive(Debug, Clone)]
pub struct TextureData<'a> {
    pub width: i32,
    pub height: i32,
    pub rgb_pixels: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct TextureId(pub i32);

// ### Bodies, Joints & Base State
//
// | API | Status | Notes |
// | --- | --- | --- |
// | getNumBodies | **Implemented** | Enumeration helper |
// | getBodyUniqueId | **Implemented** | Enumeration helper |
// | getBodyInfo | **Implemented** | Names cached |
// | computeDofCount | **Implemented** | Useful with dynamics |
// | syncBodyInfo | **Implemented** | Multi-client support |
// | syncUserData | **Implemented** | Multi-client support |
// | addUserData | **Implemented** | User data authoring |
// | getUserData | **Implemented** | User data query |
// | removeUserData | **Implemented** | User data cleanup |
// | getUserDataId | **Implemented** | User data query |
// | getNumUserData | **Implemented** | User data query |
// | getUserDataInfo | **Implemented** | User data query |
// | getBasePositionAndOrientation | **Implemented** | Uses actual state request |
// | getAABB | **Implemented** | Contact bounds |
// | resetBasePositionAndOrientation | **Implemented** | World authoring priority |
// | unsupportedChangeScaling | Optional | Rudimentary scaling |
// | getBaseVelocity | **Implemented** | Uses actual state request |
// | resetBaseVelocity | **Implemented** | Dynamics priority |
// | getNumJoints | **Implemented** | Joint enumeration |
// | getJointInfo | **Implemented** | Joint metadata |
// | getJointState | **Implemented** | Single joint sensor |
// | getJointStates | **Implemented** | Batch sensor support |
// | getJointStateMultiDof | **Implemented** | Multi-DoF sensor |
// | getJointStatesMultiDof | **Implemented** | Multi-DoF batch |
// | getLinkState | **Implemented** | Forward kinematics |
// | getLinkStates | **Implemented** | Batch link state |
// | resetJointState | **Implemented** | World authoring priority |
// | resetJointStateMultiDof | **Implemented** | Multi-DoF reset |
// | resetJointStatesMultiDof | **Implemented** | Multi-DoF batch reset |

#[derive(Debug, Clone)]
pub struct BodyInfo {
    pub base_name: String,
    pub body_name: String,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub enum BodyType {
    RigidBody = 1,
    MultiBody = 2,
    SoftBody = 3,
}

impl TryFrom<i32> for BodyType {
    type Error = BulletError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(BodyType::RigidBody),
            2 => Ok(BodyType::MultiBody),
            3 => Ok(BodyType::SoftBody),
            _ => Err(BulletError::UnknownType("internal error: Unknown BodyType")),
        }
    }
}

/// An enum to represent different types of joints
#[derive(Debug, PartialEq, Copy, Clone, Default)]
pub enum JointType {
    Revolute = 0,
    Prismatic = 1,
    Spherical = 2,
    Planar = 3,
    #[default]
    Fixed = 4,
    Point2Point = 5,
    Gear = 6,
}

impl TryFrom<i32> for JointType {
    type Error = BulletError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(JointType::Revolute),
            1 => Ok(JointType::Prismatic),
            2 => Ok(JointType::Spherical),
            3 => Ok(JointType::Planar),
            4 => Ok(JointType::Fixed),
            5 => Ok(JointType::Point2Point),
            6 => Ok(JointType::Gear),
            _ => Err(BulletError::UnknownType(
                "could not convert into a valid joint type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct JointInfo {
    pub joint_index: i32,
    pub joint_name: String,
    pub joint_type: JointType,
    pub q_index: i32,
    pub u_index: i32,
    pub q_size: i32,
    pub u_size: i32,
    pub flags: i32,
    pub damping: f64,
    pub friction: f64,
    pub lower_limit: f64,
    pub upper_limit: f64,
    pub max_force: f64,
    pub max_velocity: f64,
    pub link_name: String,
    pub joint_axis: [f64; 3],
    pub parent_frame_pos: [f64; 3],
    pub parent_frame_orn: [f64; 4],
    pub parent_index: i32,
}

impl TryFrom<b3JointInfo> for JointInfo {
    type Error = BulletError;

    fn try_from(info: b3JointInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            joint_index: info.m_joint_index,
            joint_name: PhysicsClient::read_c_string(&info.m_link_name),
            joint_type: info.m_joint_type.try_into().unwrap_or(JointType::Fixed),
            q_index: info.m_q_index,
            u_index: info.m_u_index,
            q_size: info.m_q_size,
            u_size: info.m_u_size,
            flags: info.m_flags,
            damping: info.m_joint_damping,
            friction: info.m_joint_friction,
            lower_limit: info.m_joint_lower_limit,
            upper_limit: info.m_joint_upper_limit,
            max_force: info.m_joint_max_force,
            max_velocity: info.m_joint_max_velocity,
            link_name: PhysicsClient::read_c_string(&info.m_link_name),
            joint_axis: info.m_joint_axis,
            parent_frame_pos: [
                info.m_parent_frame[0],
                info.m_parent_frame[1],
                info.m_parent_frame[2],
            ],
            parent_frame_orn: [
                info.m_parent_frame[3],
                info.m_parent_frame[4],
                info.m_parent_frame[5],
                info.m_parent_frame[6],
            ],
            parent_index: info.m_parent_index,
        })
    }
}

#[derive(Debug, Clone)]
pub struct JointState {
    pub position: f64,
    pub velocity: f64,
    pub force_torque: [f64; 6],
    pub motor_torque: f64,
}

#[derive(Debug, Clone)]
pub struct UserDataInfo {
    pub user_data_id: i32,
    pub key: String,
    pub body_unique_id: i32,
    pub link_index: i32,
    pub visual_shape_index: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct LinkState {
    pub world: na::Isometry3<f64>,
    pub local_inertial: na::Isometry3<f64>,
    pub world_link_frame: na::Isometry3<f64>,
    pub world_velocity: Option<[f64; 6]>,
    pub world_aabb: Option<Aabb>,
}

impl TryFrom<b3LinkState> for LinkState {
    type Error = BulletError;

    fn try_from(value: b3LinkState) -> Result<Self, Self::Error> {
        Ok(Self {
            world: na::Isometry3::from_parts(
                na::Translation3::from(value.m_world_position),
                na::UnitQuaternion::from_quaternion(value.m_world_orientation.into()),
            ),
            local_inertial: na::Isometry3::from_parts(
                na::Translation3::from(value.m_local_inertial_position),
                na::UnitQuaternion::from_quaternion(value.m_local_inertial_orientation.into()),
            ),
            world_link_frame: na::Isometry3::from_parts(
                na::Translation3::from(value.m_world_link_frame_position),
                na::UnitQuaternion::from_quaternion(value.m_world_link_frame_orientation.into()),
            ),
            world_velocity: if value.m_world_linear_velocity != [0.0; 3]
                || value.m_world_angular_velocity != [0.0; 3]
            {
                Some([
                    value.m_world_linear_velocity[0],
                    value.m_world_linear_velocity[1],
                    value.m_world_linear_velocity[2],
                    value.m_world_angular_velocity[0],
                    value.m_world_angular_velocity[1],
                    value.m_world_angular_velocity[2],
                ])
            } else {
                None
            },
            world_aabb: if value.m_world_linear_velocity != [0.0; 3]
                || value.m_world_angular_velocity != [0.0; 3]
            {
                Some(Aabb { min: value.m_world_aabb_min, max: value.m_world_aabb_max })
            } else {
                None
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct JointStateMultiDof {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub reaction_forces: [f64; 6],
    pub motor_torques: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MultiDofTarget<'a> {
    pub joint_index: i32,
    pub positions: Option<&'a [f64]>,
    pub velocities: Option<&'a [f64]>,
}

// ### Dynamics & Control
//
// | API | Status | Notes |
// | --- | --- | --- |
// | changeDynamics | **Implemented** | Mutation wrapper |
// | getDynamicsInfo | **Implemented** | Mirrors Bullet query |
// | setJointMotorControl | Optional | Legacy single-call (deprecated) |
// | setJointMotorControl2 | **Implemented** | Primary motor control path |
// | setJointMotorControlMultiDof | **Implemented** | Multi-DoF control |
// | setJointMotorControlMultiDofArray | **Implemented** | Multi-DoF batch control |
// | setJointMotorControlArray | **Implemented** | Batch joint control |
// | applyExternalForce | **Implemented** | Core dynamics action |
// | applyExternalTorque | **Implemented** | Core dynamics action |
// | calculateInverseDynamics | **Implemented** | Advanced dynamics |
// | calculateJacobian | **Implemented** | Advanced dynamics |
// | calculateMassMatrix | **Implemented** | Advanced dynamics |
// | calculateInverseKinematics | **Implemented** | Depends on jacobians |
// | calculateInverseKinematics2 | **Implemented** | Multi-end-effector variant |

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DynamicsInfo {
    pub mass: f64,
    pub lateral_friction: f64,
    pub local_inertia_diagonal: [f64; 3],
    pub local_inertia_position: [f64; 3],
    pub local_inertia_orientation: [f64; 4],
    pub restitution: f64,
    pub rolling_friction: f64,
    pub spinning_friction: f64,
    pub contact_damping: f64,
    pub contact_stiffness: f64,
    pub body_type: BodyType,
    pub collision_margin: f64,
    pub angular_damping: f64,
    pub linear_damping: f64,
    pub ccd_swept_sphere_radius: f64,
    pub contact_processing_threshold: f64,
    pub activation_state: i32,
    pub friction_anchor: bool,
    // TODO enumify
    pub dynamic_type: i32,
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct DynamicsUpdate {
    pub mass: Option<f64>,
    pub local_inertia_diagonal: Option<[f64; 3]>,
    pub anisotropic_friction: Option<[f64; 3]>,
    pub joint_limit: Option<(f64, f64)>,
    pub joint_limit_force: Option<f64>,
    pub dynamic_type: Option<i32>,
    pub lateral_friction: Option<f64>,
    pub spinning_friction: Option<f64>,
    pub rolling_friction: Option<f64>,
    pub restitution: Option<f64>,
    pub linear_damping: Option<f64>,
    pub angular_damping: Option<f64>,
    pub joint_damping: Option<f64>,
    pub contact_stiffness_and_damping: Option<(f64, f64)>,
    pub friction_anchor: Option<bool>,
    pub ccd_swept_sphere_radius: Option<f64>,
    pub contact_processing_threshold: Option<f64>,
    pub activation_state: Option<i32>,
    pub max_joint_velocity: Option<f64>,
    pub collision_margin: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum ControlMode {
    /// Velocity control with the desired joint velocity
    Velocity(f64),
    /// Torque control with the desired joint torque.
    Torque(f64),
    /// Position Control with the desired joint position.
    Position(f64),
    /// PD Control
    Pd {
        /// desired target position
        target_position: f64,
        /// desired target velocity
        target_velocity: f64,
        /// position gain
        position_gain: f64,
        /// velocity gain
        velocity_gain: f64,
        /// limits the velocity of a joint
        maximum_velocity: Option<f64>,
    },
    StablePd {
        /// desired target position
        target_position: f64,
        /// desired target velocity
        target_velocity: f64,
        /// position gain
        position_gain: f64,
        /// velocity gain
        velocity_gain: f64,
        /// limits the velocity of a joint
        maximum_velocity: Option<f64>,
    },
}

impl ControlMode {
    pub fn as_raw(self) -> i32 {
        match self {
            ControlMode::Velocity(_) => 0,
            ControlMode::Torque(_) => 1,
            ControlMode::Position(_) => 2,
            ControlMode::Pd { .. } => 3,
            ControlMode::StablePd { .. } => 4,
        }
    }

    pub fn is_position_based(self) -> bool {
        matches!(
            self,
            ControlMode::Position(_) | ControlMode::Pd { .. } | ControlMode::StablePd { .. }
        )
    }

    pub fn is_velocity_based(self) -> bool {
        matches!(self, ControlMode::Velocity(_))
    }

    pub fn is_torque_based(self) -> bool {
        matches!(self, ControlMode::Torque(_))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ControlModeArray<'a> {
    /// Velocity control with the desired joint velocity
    Velocity(&'a [f64]),
    /// Torque control with the desired joint torque.
    Torque(&'a [f64]),
    /// Position Control with the desired joint position.
    Position(&'a [f64]),
    /// PD Control
    Pd {
        /// desired target position
        target_position: &'a [f64],
        /// desired target velocity
        target_velocity: &'a [f64],
        /// position gain
        position_gain: &'a [f64],
        /// velocity gain
        velocity_gain: &'a [f64],
        /// limits the velocity of a joint
        maximum_velocity: Option<&'a [f64]>,
    },
    StablePd {
        /// desired target position
        target_position: &'a [f64],
        /// desired target velocity
        target_velocity: &'a [f64],
        /// position gain
        position_gain: &'a [f64],
        /// velocity gain
        velocity_gain: &'a [f64],
        /// limits the velocity of a joint
        maximum_velocity: Option<&'a [f64]>,
    },
}

impl ControlModeArray<'_> {
    pub fn as_raw(self) -> i32 {
        match self {
            ControlModeArray::Velocity(_) => 0,
            ControlModeArray::Torque(_) => 1,
            ControlModeArray::Position(_) => 2,
            ControlModeArray::Pd { .. } => 3,
            ControlModeArray::StablePd { .. } => 4,
        }
    }
}

/// Multi-DoF 单关节控制参数，使用枚举携带模式与数据（对齐 `set_joint_motor_control` 的风格）。
#[derive(Debug, Clone)]
pub enum MultiDofControl<'a> {
    /// 位置/PD/Stable-PD 类控制，提供位置与速度目标、增益、力等。
    Position {
        target_positions: Option<&'a [f64]>,
        target_velocities: Option<&'a [f64]>,
        position_gains: Option<&'a [f64]>,
        velocity_gains: Option<&'a [f64]>,
        forces: Option<&'a [f64]>,
        damping: Option<&'a [f64]>,
        /// 限制最大速度（与 PD/StablePD 一致，可选）
        max_velocity: Option<f64>,
    },
    /// PD 控制（行为与 Position 分支一致，仅控制码不同）。
    Pd {
        target_positions: Option<&'a [f64]>,
        target_velocities: Option<&'a [f64]>,
        position_gains: Option<&'a [f64]>,
        velocity_gains: Option<&'a [f64]>,
        forces: Option<&'a [f64]>,
        damping: Option<&'a [f64]>,
        max_velocity: Option<f64>,
    },
    /// Stable-PD 控制（行为与 Position 分支一致，仅控制码不同）。
    StablePd {
        target_positions: Option<&'a [f64]>,
        target_velocities: Option<&'a [f64]>,
        position_gains: Option<&'a [f64]>,
        velocity_gains: Option<&'a [f64]>,
        forces: Option<&'a [f64]>,
        damping: Option<&'a [f64]>,
        max_velocity: Option<f64>,
    },
    /// 速度控制，提供速度目标、速度增益、以及力限制。
    Velocity {
        target_velocities: Option<&'a [f64]>,
        velocity_gains: Option<&'a [f64]>,
        forces: Option<&'a [f64]>,
        damping: Option<&'a [f64]>,
    },
    /// 力/力矩控制，直接指定力数组。
    Torque {
        forces: Option<&'a [f64]>,
        damping: Option<&'a [f64]>,
    },
}

impl MultiDofControl<'_> {
    /// 与底层 Bullet 控制码保持一致，用于初始化命令。
    pub fn as_raw(&self) -> i32 {
        match self {
            MultiDofControl::Velocity { .. } => 0,
            MultiDofControl::Torque { .. } => 1,
            MultiDofControl::Position { .. } => 2,
            MultiDofControl::Pd { .. } => 3,
            MultiDofControl::StablePd { .. } => 4,
        }
    }

    pub fn is_position_based(&self) -> bool {
        matches!(
            self,
            MultiDofControl::Position { .. }
                | MultiDofControl::Pd { .. }
                | MultiDofControl::StablePd { .. }
        )
    }
    pub fn is_velocity_based(&self) -> bool {
        matches!(self, MultiDofControl::Velocity { .. })
    }
    pub fn is_torque_based(&self) -> bool {
        matches!(self, MultiDofControl::Torque { .. })
    }
}

/// Multi-DoF 批量控制条目：关联关节索引与控制枚举。
#[derive(Debug, Clone)]
pub struct MultiDofControlEntry<'a> {
    pub joint_index: i32,
    pub control: MultiDofControl<'a>,
}

// 旧的 JointMotorControlMultiDofOptions / JointMotorControlMultiDofArrayOptions 已移除，
// 请使用 MultiDofControl 与 MultiDofControlEntry。

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum ForceFrame {
    Link,
    World,
}

impl ForceFrame {
    #[allow(dead_code)]
    pub fn as_raw(self) -> i32 {
        match self {
            ForceFrame::Link => 1,
            ForceFrame::World => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Jacobian {
    pub linear: na::Matrix3xX<f64>,
    pub angular: na::Matrix3xX<f64>,
}

#[derive(Debug, Clone)]
pub struct MassMatrix {
    pub matrix: na::DMatrix<f64>,
}

impl Jacobian {
    pub fn zeros(columns: usize) -> Self {
        Self {
            linear: na::Matrix3xX::zeros(columns),
            angular: na::Matrix3xX::zeros(columns),
        }
    }
}

impl MassMatrix {
    pub fn as_matrix(&self) -> &na::DMatrix<f64> {
        &self.matrix
    }
}

impl std::ops::Deref for MassMatrix {
    type Target = na::DMatrix<f64>;

    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl std::ops::DerefMut for MassMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.matrix
    }
}

/// Specifies which Inverse Kinematics Solver to use in
/// [`calculate_inverse_kinematics()`](`crate::client::PhysicsClient::calculate_inverse_kinematics()`)
#[derive(Debug, Default, Clone, Copy)]
pub enum IkSolver {
    /// Damped Least Squares
    #[default]
    Dls = 0,
    /// Selective Damped Least
    Sdls = 1,
}

#[derive(Debug, Clone, Default)]
pub struct InverseKinematicsOptions<'a> {
    pub joint_damping: Option<&'a [f64]>,
    pub current_positions: Option<&'a [f64]>,
    pub solver: IkSolver,
    pub max_iterations: Option<i32>,
    pub residual_threshold: Option<f64>,

    pub lower_limits: Option<&'a [f64]>,
    pub upper_limits: Option<&'a [f64]>,
    pub joint_ranges: Option<&'a [f64]>,
    pub rest_poses: Option<&'a [f64]>,
}

#[derive(Debug, Clone)]
pub struct InverseKinematicsMultiTargetOptions<'a> {
    pub end_effector_link_indices: &'a [i32],
    pub target_positions: &'a [[f64; 3]],
    pub lower_limits: Option<&'a [f64]>,
    pub upper_limits: Option<&'a [f64]>,
    pub joint_ranges: Option<&'a [f64]>,
    pub rest_poses: Option<&'a [f64]>,
    pub joint_damping: Option<&'a [f64]>,
    pub current_positions: Option<&'a [f64]>,
    pub solver: Option<i32>,
    pub max_iterations: Option<i32>,
    pub residual_threshold: Option<f64>,
}

// ### Collision Queries & Contact Data
//
// | API | Status | Notes |
// | --- | --- | --- |
// | getContactPoints | **Implemented** | Returns per-contact data |
// | getClosestPoints | **Implemented** | Distance queries |
// | getOverlappingObjects | **Implemented** | AABB queries |
// | setCollisionFilterPair | **Implemented** | Collision filtering |
// | setCollisionFilterGroupMask | **Implemented** | Collision filtering |
// | rayTest | **Implemented** | Single raycast |
// | rayTestBatch | **Implemented** | Batch raycasts |

#[derive(Debug, Clone)]
pub struct ContactPoint {
    pub contact_flags: i32,
    pub body_a: i32,
    pub body_b: i32,
    pub link_a: i32,
    pub link_b: i32,
    pub position_on_a: [f64; 3],
    pub position_on_b: [f64; 3],
    pub contact_normal_on_b: [f64; 3],
    pub contact_distance: f64,
    pub normal_force: f64,
    pub linear_friction_force_1: f64,
    pub linear_friction_direction_1: [f64; 3],
    pub linear_friction_force_2: f64,
    pub linear_friction_direction_2: [f64; 3],
}

pub type ClosestPoint = ContactPoint;

#[derive(Debug, Clone, Copy)]
pub struct OverlappingObject {
    pub object_unique_id: i32,
    pub link_index: i32,
}

#[derive(Debug, Clone)]
pub struct RayHit {
    pub object_unique_id: i32,
    pub link_index: i32,
    pub hit_fraction: f64,
    pub hit_position_world: [f64; 3],
    pub hit_normal_world: [f64; 3],
}

#[derive(Debug, Clone, Default)]
pub struct ClosestPointsOptions {
    pub link_index_a: Option<i32>,
    pub link_index_b: Option<i32>,
    pub collision_shape_a: Option<i32>,
    pub collision_shape_b: Option<i32>,
    pub collision_shape_position_a: Option<[f64; 3]>,
    pub collision_shape_position_b: Option<[f64; 3]>,
    pub collision_shape_orientation_a: Option<[f64; 4]>,
    pub collision_shape_orientation_b: Option<[f64; 4]>,
}

#[derive(Debug, Default, Clone)]
pub struct RayTestBatchOptions<'a> {
    pub num_threads: Option<i32>,
    pub parent_object_unique_id: Option<i32>,
    pub parent_link_index: Option<i32>,
    pub report_hit_number: Option<i32>,
    pub collision_filter_mask: Option<i32>,
    pub fraction_epsilon: Option<f64>,
    pub ray_from_positions: &'a [[f64; 3]],
    pub ray_to_positions: &'a [[f64; 3]],
}

// ### Rendering, Debug & Visualization
//
// | API | Status | Notes |
// | --- | --- | --- |
// | renderImage | Pending | Obsolete; low priority |
// | getCameraImage | Implemented | High effort (buffers) |
// | isNumpyEnabled | Pending | Simple flag query |
// | computeViewMatrix | Implemented | Math helper |
// | computeViewMatrixFromYawPitchRoll | Implemented | Math helper |
// | computeProjectionMatrix | Implemented | Math helper |
// | computeProjectionMatrixFOV | Implemented | Math helper |
// | addUserDebugLine | Implemented | Debug draw |
// | addUserDebugPoints | Implemented | Debug draw |
// | addUserDebugText | Implemented | Debug draw |
// | addUserDebugParameter | Implemented | GUI slider/button |
// | readUserDebugParameter | Implemented | GUI feedback |
// | removeUserDebugItem | Implemented | Debug cleanup |
// | removeAllUserDebugItems | Implemented | Debug cleanup |
// | removeAllUserParameters | Implemented | Debug cleanup |
// | setDebugObjectColor | Implemented | Debug override |
// | getDebugVisualizerCamera | Implemented | Visualizer query |
// | configureDebugVisualizer | Implemented | Visualizer toggle |
// | resetDebugVisualizerCamera | Implemented | Visualizer camera |
// | getVisualShapeData | Implemented | Visual query |
// | getCollisionShapeData | Implemented | Collision query |

#[derive(Debug, Clone)]
pub struct CameraImage {
    pub width: i32,
    pub height: i32,
    pub rgba: Vec<u8>,
    pub depth: Vec<f32>,
    pub segmentation_mask: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
pub enum Renderer {
    TinyRenderer = 1 << 16,
    /// Direct mode has no OpenGL, so you can not use this setting in direct mode.
    BulletHardwareOpenGl = 1 << 17,
}

#[derive(Debug, Clone, Default)]
pub struct CameraImageOptions {
    pub width: i32,
    pub height: i32,
    pub view_matrix: Option<[f32; 16]>,
    pub projection_matrix: Option<[f32; 16]>,
    pub light_direction: Option<[f32; 3]>,
    pub light_color: Option<[f32; 3]>,
    pub light_distance: Option<f32>,
    pub shadow: Option<bool>,
    pub light_ambient_coeff: Option<f32>,
    pub light_diffuse_coeff: Option<f32>,
    pub light_specular_coeff: Option<f32>,
    pub renderer: Option<Renderer>,
    pub flags: Option<i32>,
    pub projective_texture_view: Option<[f32; 16]>,
    pub projective_texture_projection: Option<[f32; 16]>,
}

impl CameraImageOptions {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height, ..Self::default() }
    }
}

#[derive(Debug, Clone)]
pub struct DebugLineOptions {
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub color: Option<[f64; 3]>,
    pub line_width: f64,
    pub life_time: f64,
    pub parent_object_unique_id: Option<i32>,
    pub parent_link_index: Option<i32>,
    pub replace_item_unique_id: Option<i32>,
}

impl Default for DebugLineOptions {
    fn default() -> Self {
        Self {
            from: [0.0; 3],
            to: [0.0, 0.0, 1.0],
            color: None,
            line_width: 1.0,
            life_time: 0.0,
            parent_object_unique_id: None,
            parent_link_index: None,
            replace_item_unique_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugPointsOptions<'a> {
    pub positions: &'a [[f64; 3]],
    pub colors: &'a [[f64; 3]],
    pub point_size: f64,
    pub life_time: f64,
    pub parent_object_unique_id: Option<i32>,
    pub parent_link_index: Option<i32>,
    pub replace_item_unique_id: Option<i32>,
}

impl Default for DebugPointsOptions<'_> {
    fn default() -> Self {
        static EMPTY: [[f64; 3]; 0] = [];
        Self {
            positions: &EMPTY,
            colors: &EMPTY,
            point_size: 1.0,
            life_time: 0.0,
            parent_object_unique_id: None,
            parent_link_index: None,
            replace_item_unique_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugTextOptions<'a> {
    pub text: &'a str,
    pub position: [f64; 3],
    pub color: Option<[f64; 3]>,
    pub size: f64,
    pub life_time: f64,
    pub orientation: Option<[f64; 4]>,
    pub parent_object_unique_id: Option<i32>,
    pub parent_link_index: Option<i32>,
    pub replace_item_unique_id: Option<i32>,
}

impl Default for DebugTextOptions<'_> {
    fn default() -> Self {
        Self {
            text: "",
            position: [0.0; 3],
            color: None,
            size: 1.0,
            life_time: 0.0,
            orientation: None,
            parent_object_unique_id: None,
            parent_link_index: None,
            replace_item_unique_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugParameterOptions<'a> {
    pub name: &'a str,
    pub range_min: f64,
    pub range_max: f64,
    pub start_value: f64,
}

impl Default for DebugParameterOptions<'_> {
    fn default() -> Self {
        Self { name: "", range_min: 0.0, range_max: 1.0, start_value: 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct DebugVisualizerCamera {
    pub width: i32,
    pub height: i32,
    pub view_matrix: [f32; 16],
    pub projection_matrix: [f32; 16],
    pub camera_up: [f32; 3],
    pub camera_forward: [f32; 3],
    pub horizontal: [f32; 3],
    pub vertical: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
pub enum DebugVisualizerFlag {
    CovEnableGui = 1,
    CovEnableShadows,
    CovEnableWireframe,
    CovEnableVrTeleporting,
    CovEnableVrPicking,
    CovEnableVrRenderControllers,
    CovEnableRendering,
    CovEnableSyncRenderingInternal,
    CovEnableKeyboardShortcuts,
    CovEnableMousePicking,
    CovEnableYAxisUp,
    CovEnableTinyRenderer,
    CovEnableRgbBufferPreview,
    CovEnableDepthBufferPreview,
    CovEnableSegmentationMarkPreview,
    CovEnablePlanarReflection,
    CovEnableSingleStepRendering,
}

#[derive(Debug, Clone)]
pub enum DebugVisualizerOptions {
    Flag(DebugVisualizerFlag, bool),
    LightPosition([f32; 3]),
    ShadowMapResolution(i32),
    ShadowMapWorldSize(i32),
    RemoteSyncTransformInterval(f64),
    ShadowMapIntensity(f64),
    RgbBackground([f32; 3]),
}

#[derive(Debug, Clone)]
pub struct ResetDebugVisualizerCameraOptions {
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub target: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct VisualShapeData {
    pub object_unique_id: i32,
    pub link_index: i32,
    pub geometry_type: i32,
    pub dimensions: [f64; 3],
    pub mesh_asset_file_name: String,
    pub local_visual_frame_position: [f64; 3],
    pub local_visual_frame_orientation: [f64; 4],
    pub rgba_color: [f64; 4],
    pub tiny_renderer_texture_id: i32,
    pub texture_unique_id: i32,
    pub opengl_texture_id: i32,
}

#[derive(Debug, Clone)]
pub struct CollisionShapeData {
    pub object_unique_id: i32,
    pub link_index: i32,
    pub geometry_type: i32,
    pub dimensions: [f64; 3],
    pub mesh_asset_file_name: String,
    pub local_collision_frame_position: [f64; 3],
    pub local_collision_frame_orientation: [f64; 4],
}

// =================== VR, Input, Logging, Plugin Types ===================

pub const VR_EVENT_FLAG_CONTROLLER_MOVE: i32 = ffi::VR_CONTROLLER_MOVE_EVENT;
pub const VR_EVENT_FLAG_CONTROLLER_BUTTON: i32 = ffi::VR_CONTROLLER_BUTTON_EVENT;
pub const VR_EVENT_FLAG_HMD_MOVE: i32 = ffi::VR_HMD_MOVE_EVENT;
pub const VR_EVENT_FLAG_GENERIC_TRACKER_MOVE: i32 = ffi::VR_GENERIC_TRACKER_MOVE_EVENT;
pub const VR_DEVICE_FILTER_CONTROLLER: i32 = ffi::VR_DEVICE_CONTROLLER;
pub const VR_DEVICE_FILTER_HMD: i32 = ffi::VR_DEVICE_HMD;
pub const VR_DEVICE_FILTER_GENERIC_TRACKER: i32 = ffi::VR_DEVICE_GENERIC_TRACKER;
pub const VR_BUTTON_IS_DOWN: i32 = 1;
pub const VR_BUTTON_TRIGGERED: i32 = 2;
pub const VR_BUTTON_RELEASED: i32 = 4;
pub const VR_CAMERA_TRACK_OBJECT_ORIENTATION: i32 = ffi::VR_CAMERA_TRACK_OBJECT_ORIENTATION;
pub const VR_MAX_ANALOG_AXIS: usize = ffi::MAX_VR_ANALOG_AXIS;
pub const VR_MAX_BUTTONS: usize = ffi::MAX_VR_BUTTONS;

#[derive(Debug, Clone)]
pub struct VrControllerEvent {
    pub controller_id: i32,
    pub device_type: i32,
    pub num_move_events: i32,
    pub num_button_events: i32,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub analog_axis: f32,
    pub aux_analog_axis: [f32; VR_MAX_ANALOG_AXIS * 2],
    pub buttons: [i32; VR_MAX_BUTTONS],
}

impl Default for VrControllerEvent {
    fn default() -> Self {
        Self {
            controller_id: 0,
            device_type: 0,
            num_move_events: 0,
            num_button_events: 0,
            position: [0.0; 3],
            orientation: [0.0; 4],
            analog_axis: 0.0,
            aux_analog_axis: [0.0; VR_MAX_ANALOG_AXIS * 2],
            buttons: [0; VR_MAX_BUTTONS],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VrEventsOptions {
    pub device_type_filter: Option<i32>,
    pub include_aux_analog_axes: bool,
}

#[derive(Debug, Clone, Default)]
pub struct VrCameraStateOptions {
    pub root_position: Option<[f64; 3]>,
    pub root_orientation: Option<[f64; 4]>,
    pub tracking_object_unique_id: Option<i32>,
    pub tracking_flag: Option<i32>,
}

/// Represents a key press Event
#[derive(Debug, Copy, Clone, Default)]
pub struct KeyboardEvent {
    /// specifies which key the event is about.
    pub key: char,
    pub(crate) key_state: i32,
}

impl KeyboardEvent {
    /// is true when the key goes from an "up" to a "down" state.
    pub fn was_triggered(&self) -> bool {
        self.key_state & 2 == 2
    }
    /// is true when the key is currently pressed.
    pub fn is_down(&self) -> bool {
        self.key_state & 1 == 1
    }
    /// is true when the key goes from a "down" to an "up" state.
    pub fn is_released(&self) -> bool {
        self.key_state & 4 == 4
    }
}
impl From<b3KeyboardEvent> for KeyboardEvent {
    fn from(event: b3KeyboardEvent) -> Self {
        Self {
            key: std::char::from_u32(event.m_keyCode as u32).unwrap_or('\0'),
            key_state: event.m_keyState,
        }
    }
}

/// Mouse Events can either be a "Move" or a "Button" event. A "Move" event is when the mouse is moved
/// in the OpenGL window and a "Button" even is when a mouse button is clicked.
#[derive(Debug, Copy, Clone)]
pub enum MouseEvent {
    /// Contains the mouse position
    Move {
        /// x-coordinate of the mouse pointer
        mouse_x: f32,
        /// y-coordinate of the mouse pointer
        mouse_y: f32,
    },
    /// Specifies Mouse Position and a Button event
    Button {
        /// x-coordinate of the mouse pointer
        mouse_x: f32,
        /// y-coordinate of the mouse pointer
        mouse_y: f32,
        /// button index for left/middle/right mouse button
        button_index: i32,
        /// state of the mouse button
        button_state: MouseButtonState,
    },
}

/// Represents the different possible states of a mouse button
#[derive(Debug, Copy, Clone)]
pub struct MouseButtonState {
    pub(crate) flag: i32,
}
impl MouseButtonState {
    /// is true when the button goes from an "unpressed" to a "pressed" state.
    pub fn was_triggered(&self) -> bool {
        self.flag & 2 == 2
    }
    /// is true when the button is in a "pressed" state.
    pub fn is_pressed(&self) -> bool {
        self.flag & 1 == 1
    }
    /// is true when the button goes from a "pressed" to an "unpressed" state.
    pub fn is_released(&self) -> bool {
        self.flag & 4 == 4
    }
}

impl From<b3MouseEvent> for MouseEvent {
    fn from(event: b3MouseEvent) -> Self {
        if event.m_eventType == 0 {
            Self::Move { mouse_x: event.m_mousePosX, mouse_y: event.m_mousePosY }
        } else {
            Self::Button {
                mouse_x: event.m_mousePosX,
                mouse_y: event.m_mousePosY,
                button_index: event.m_buttonIndex,
                button_state: MouseButtonState { flag: event.m_buttonState },
            }
        }
    }
}

pub enum LoggingType {
    /// This will require to load the quadruped/quadruped.urdf and object unique
    /// id from the quadruped. It logs the timestamp, IMU roll/pitch/yaw, 8 leg
    /// motor positions (q0-q7), 8 leg motor torques (u0-u7), the forward speed of the
    /// torso and mode (unused in simulation).
    Minitaur = 0,
    /// This will log a log of the data of either all objects or selected ones
    /// (if [`object_ids`](`crate::types::StateLoggingOptions::object_ids`) in the
    /// [`StateLoggingOptions`](`crate::types::StateLoggingOptions`) is not empty).
    GenericRobot,
    VrControllers,
    /// this will open an MP4 file and start streaming the OpenGL 3D visualizer pixels to the file
    /// using an ffmpeg pipe. It will require ffmpeg installed. You can also use
    /// avconv (default on Ubuntu), just create a symbolic link so that ffmpeg points to avconv.
    /// On Windows, ffmpeg has some issues that cause tearing/color artifacts in some cases.
    VideoMp4,
    Commands,
    ContactPoints,
    /// This will dump a timings file in JSON format that can be opened using Google Chrome <about://tracing> LOAD.
    ProfileTimings,
    AllCommands,
    ReplayAllCommands,
    CustomTimer,
}
bitflags::bitflags! {
    #[derive(Debug, Clone)]
    pub struct LogFlags : i32 {
        const JOINT_MOTOR_TORQUES = 1;
        const JOINT_USER_TORQUES = 2;
        const JOINT_TORQUES = 3;
    }
}

#[derive(Debug, Clone, Default)]
pub struct StateLoggingOptions {
    pub object_unique_ids: Option<Vec<i32>>,
    pub max_log_dof: Option<i32>,
    pub body_unique_id_a: Option<i32>,
    pub body_unique_id_b: Option<i32>,
    pub link_index_a: Option<i32>,
    pub link_index_b: Option<i32>,
    pub device_type_filter: Option<i32>,
    pub log_flags: Option<LogFlags>,
}

impl From<()> for StateLoggingOptions {
    fn from((): ()) -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct PluginCommandOptions<'a> {
    pub plugin_unique_id: i32,
    pub text_argument: Option<&'a str>,
    pub int_args: Option<&'a [i32]>,
    pub float_args: Option<&'a [f32]>,
}

#[derive(Debug, Clone)]
pub struct PluginCommandReturnData {
    pub value_type: i32,
    pub data: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct PluginCommandResult {
    pub status: i32,
    pub return_data: Option<PluginCommandReturnData>,
}
