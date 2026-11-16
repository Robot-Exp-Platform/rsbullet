use anyhow::Ok;
use roplat_world::{Collision, Entity, Visual};
use rsbullet_core::{
    CollisionId, CollisionShapeOptions, MultiBodyBase, MultiBodyCreateOptions, VisualId,
    VisualShapeOptions,
};

use crate::RsBullet;

pub struct CollisionMarker;
pub struct VisualMarker;

/// What [`EntityBuilder::load`](roplat_world::EntityBuilder::load) made: the
/// shape, and the static body Bullet built of it. The body's id is the one
/// a segmentation mask reports for the entity's pixels and the one body
/// queries (`get_base_frame_pose`, `get_aabb`) take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoadedEntity<S> {
    pub shape: S,
    pub body: i32,
}

/// Builds a shape as a static body of its own. `base` is the body's pose
/// in the world; the shape sits at the body's origin.
pub struct EntityBuilder<'a, T> {
    _marker: T,
    pub(crate) rsbullet: &'a mut RsBullet,
    entity: Entity<'a>,
    name: Option<String>,
    base: Option<nalgebra::Isometry3<f64>>,
    base_fixed: Option<bool>,
    scaling: Option<f64>,
}

impl<'a> EntityBuilder<'a, CollisionMarker> {
    pub fn collision(rsbullet: &'a mut RsBullet, collision: Collision<'a>) -> Self {
        Self {
            _marker: CollisionMarker,
            rsbullet,
            entity: collision,
            name: None,
            base: None,
            base_fixed: None,
            scaling: None,
        }
    }
}

impl<'a> EntityBuilder<'a, VisualMarker> {
    pub fn visual(rsbullet: &'a mut RsBullet, visual: Visual<'a>) -> Self {
        Self {
            _marker: VisualMarker,
            rsbullet,
            entity: visual,
            name: None,
            base: None,
            base_fixed: None,
            scaling: None,
        }
    }
}

impl<'a> roplat_world::EntityBuilder<'a> for EntityBuilder<'a, CollisionMarker> {
    type Entity = LoadedEntity<CollisionId>;
    fn name(mut self, _name: String) -> Self {
        self.name = Some(_name);
        self
    }
    fn base(mut self, base: impl Into<nalgebra::Isometry3<f64>>) -> Self {
        self.base = Some(base.into());
        self
    }
    fn base_fixed(mut self, base_fixed: bool) -> Self {
        self.base_fixed = Some(base_fixed);
        self
    }
    fn scaling(mut self, scaling: f64) -> Self {
        self.scaling = Some(scaling);
        self
    }
    fn load(self) -> anyhow::Result<Self::Entity> {
        // The shape at the body's origin: the base places the body, once.
        let shape = self
            .rsbullet
            .client
            .create_collision_shape(&self.entity.into(), None::<CollisionShapeOptions>)
            .map(CollisionId)?;

        let body = self
            .rsbullet
            .client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    collision_shape: shape,
                    pose: self.base.unwrap_or_default(),
                    ..Default::default()
                },
                ..Default::default()
            })?;

        Ok(LoadedEntity { shape, body })
    }
}

impl<'a> roplat_world::EntityBuilder<'a> for EntityBuilder<'a, VisualMarker> {
    type Entity = LoadedEntity<VisualId>;
    fn name(mut self, _name: String) -> Self {
        self.name = Some(_name);
        self
    }
    fn base(mut self, base: impl Into<nalgebra::Isometry3<f64>>) -> Self {
        self.base = Some(base.into());
        self
    }
    fn base_fixed(mut self, base_fixed: bool) -> Self {
        self.base_fixed = Some(base_fixed);
        self
    }
    fn scaling(mut self, scaling: f64) -> Self {
        self.scaling = Some(scaling);
        self
    }
    fn load(self) -> anyhow::Result<Self::Entity> {
        // The shape at the body's origin: the base places the body, once.
        let shape = self
            .rsbullet
            .client
            .create_visual_shape(&self.entity.into(), None::<VisualShapeOptions>)
            .map(VisualId)?;

        let body = self
            .rsbullet
            .client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    visual_shape: shape,
                    pose: self.base.unwrap_or_default(),
                    ..Default::default()
                },
                ..Default::default()
            })?;

        Ok(LoadedEntity { shape, body })
    }
}
