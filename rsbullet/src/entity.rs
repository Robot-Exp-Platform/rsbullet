use anyhow::Ok;
use roplat_behavior::{Collision, Entity, Visual};
use rsbullet_core::{CollisionId, MultiBodyBase, MultiBodyCreateOptions, VisualId};

use crate::RsBullet;

pub struct CollisionMarker;
pub struct VisualMarker;

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

impl<'a> roplat_behavior::EntityBuilder<'a> for EntityBuilder<'a, CollisionMarker> {
    type Entity = CollisionId;
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
        let id = self
            .rsbullet
            .client
            .create_collision_shape(&self.entity.into(), self.base)
            .map(CollisionId)?;

        self.rsbullet
            .client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    collision_shape: id,
                    pose: self.base.unwrap_or_default(),
                    ..Default::default()
                },
                ..Default::default()
            })?;

        Ok(id)
    }
}

impl<'a> roplat_behavior::EntityBuilder<'a> for EntityBuilder<'a, VisualMarker> {
    type Entity = VisualId;
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
        let id = self
            .rsbullet
            .client
            .create_visual_shape(&self.entity.into(), self.base)
            .map(VisualId)?;

        self.rsbullet
            .client
            .create_multi_body(&MultiBodyCreateOptions {
                base: MultiBodyBase {
                    mass: 0.,
                    visual_shape: id,
                    pose: self.base.unwrap_or_default(),
                    ..Default::default()
                },
                ..Default::default()
            })?;

        Ok(id)
    }
}
