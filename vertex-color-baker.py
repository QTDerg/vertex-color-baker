bl_info = {
    "name": "Vertex Color Baker",
    "version": (2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Vertex Paint",
    "description": "Bake scene lighting to vertex colors using Cycles (GPU)",
    "category": "Paint",
}

import bpy
import numpy as np
import time

class MESH_OT_gpu_vertex_paint(bpy.types.Operator):
    """Bake scene lighting to vertex colors using GPU/Cycles"""
    bl_idname = "mesh.gpu_vertex_paint"
    bl_label = "Bake Lighting (GPU)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        start_time = time.time()
        obj = context.active_object
        props = context.scene.adv_vertex_paint_props
        
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}

        # 1. STORE CURRENT STATE
        # We need to restore these settings after baking
        prev_engine = context.scene.render.engine
        prev_device = context.scene.cycles.device
        prev_samples = context.scene.cycles.samples
        prev_preview_samples = context.scene.cycles.preview_samples
        prev_bake_type = context.scene.cycles.bake_type
        prev_bake_target = context.scene.render.bake.target
        
        # Store original materials to restore later
        original_materials = [s.material for s in obj.material_slots]
        
        try:
            # 2. SETUP CYCLES & GPU
            context.scene.render.engine = 'CYCLES'
            context.scene.cycles.device = 'GPU'
            context.scene.cycles.samples = props.samples
            context.scene.cycles.preview_samples = props.samples
            
            # Attempt to set OptiX, fall back to CUDA if needed
            prefs = context.preferences.addons['cycles'].preferences
            try:
                prefs.compute_device_type = 'OPTIX'
            except:
                prefs.compute_device_type = 'CUDA'
            
            # Ensure devices are active
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True

            # 3. PREPARE MESH DATA
            # Create or get vertex color layer
            if not obj.data.color_attributes:
                obj.data.color_attributes.new(name="Baked_Lighting", type='BYTE_COLOR', domain='CORNER')
            
            # Ensure we are using the active one
            color_layer = obj.data.color_attributes.active_color
            color_layer_name = color_layer.name

            # 4. SETUP OVERRIDE MATERIAL
            # We want to bake PURE light, so we need a pure white matte material
            # This ignores the object's current textures so we only capture shadows/light
            temp_mat = bpy.data.materials.new(name="_Temp_Bake_Mat_")
            temp_mat.use_nodes = True
            nodes = temp_mat.node_tree.nodes
            nodes.clear()
            
            shader = nodes.new('ShaderNodeBsdfPrincipled')
            shader.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            shader.inputs['Roughness'].default_value = 1.0  # Diffuse look
            # Handle Blender 4.0+ Specular (removes reflection for pure diffuse bake)
            if 'Specular IOR Level' in shader.inputs:
                shader.inputs['Specular IOR Level'].default_value = 0.0
            elif 'Specular' in shader.inputs:
                shader.inputs['Specular'].default_value = 0.0
                
            output = nodes.new('ShaderNodeOutputMaterial')
            temp_mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])

            # Assign temp material to all slots
            if len(obj.material_slots) == 0:
                obj.data.materials.append(temp_mat)
            else:
                for i in range(len(obj.material_slots)):
                    obj.material_slots[i].material = temp_mat

            # 5. BAKE
            self.report({'INFO'}, "Baking lighting... (Check Progress Bar)")
            
            # Set bake settings
            # We bake 'DIFFUSE' with only Direct and Indirect light (No Color contribution)
            context.scene.cycles.bake_type = 'DIFFUSE'
            context.scene.render.bake.use_pass_direct = True
            context.scene.render.bake.use_pass_indirect = True
            context.scene.render.bake.use_pass_color = False # Don't bake the white color itself
            context.scene.render.bake.target = 'VERTEX_COLORS'
            
            # Execute Bake
            bpy.ops.object.bake(type='DIFFUSE')

            # 6. APPLY SHADOW COLOR (Post-Processing)
            # The bake resulted in physical light values (Black = Shadow, Bright = Light)
            # We now mix the User's Shadow Color into the result.
            # Logic: Final = ShadowColor + BakedLight
            
            # Access raw data using Numpy for speed (instant vs slow python loops)
            layer = obj.data.color_attributes[color_layer_name]
            
            # Get vertex colors count
            count = len(layer.data)
            
            # Create empty numpy array flattened (count * 4 for RGBA)
            # Note: attributes can be float or byte. In Blender 4+ usually float for bake.
            # We handle generic access.
            raw_colors = np.zeros(count * 4, dtype=np.float32)
            layer.data.foreach_get("color", raw_colors)
            
            # Reshape to (N, 4)
            raw_colors = raw_colors.reshape((-1, 4))
            
            # Extract RGB
            baked_rgb = raw_colors[:, :3]
            
            # Get user shadow color
            shadow_rgb = np.array([props.shadow_color[0], props.shadow_color[1], props.shadow_color[2]], dtype=np.float32)
            
            # Apply Formula: Base Shadow + (Light * Intensity)
            # This ensures dark areas get the shadow color, lit areas get (Shadow + Light)
            # You can change logic to 'mix' if you prefer, but Additive is physically accurate 
            # for "Ambient Term + Diffuse Term".
            final_rgb = shadow_rgb + (baked_rgb * props.intensity)
            
            # Clamp 0-1
            final_rgb = np.clip(final_rgb, 0.0, 1.0)
            
            # Put back into the array (preserve Alpha as 1.0)
            raw_colors[:, :3] = final_rgb
            raw_colors[:, 3] = 1.0 
            
            # Write back to Blender
            layer.data.foreach_set("color", raw_colors.flatten())

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
            
        finally:
            # 7. CLEANUP / RESTORE STATE
            
            # Restore materials
            # Resize slots if needed (simple restoration)
            if len(original_materials) > 0:
                for i, mat in enumerate(original_materials):
                    if i < len(obj.material_slots):
                        obj.material_slots[i].material = mat
            else:
                obj.data.materials.clear()
            
            # Remove temp material
            if 'temp_mat' in locals() and temp_mat:
                bpy.data.materials.remove(temp_mat)

            # Restore Render Settings
            context.scene.render.engine = prev_engine
            context.scene.cycles.device = prev_device
            context.scene.cycles.samples = prev_samples
            context.scene.cycles.preview_samples = prev_preview_samples
            context.scene.cycles.bake_type = prev_bake_type
            context.scene.render.bake.target = prev_bake_target
            
            # Switch to Vertex Paint mode
            #if context.mode != 'PAINT_VERTEX':
            #    bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        self.report({'INFO'}, f"Baked in {time.time() - start_time:.2f}s")
        return {'FINISHED'}

class AdvVertexPaintProperties(bpy.types.PropertyGroup):
    shadow_color: bpy.props.FloatVectorProperty(
        name="Shadow Color",
        description="Base ambient color (fill for shadows)",
        default=(0.05, 0.05, 0.2),
        min=0.0,
        max=1.0,
        subtype='COLOR',
        size=3,
    )
    
    intensity: bpy.props.FloatProperty(
        name="Light Intensity",
        description="Multiplier for the baked light",
        default=1.0,
        min=0.0,
        max=5.0,
    )
    
    samples: bpy.props.IntProperty(
        name="Bake Samples",
        description="Higher values = less noise, slower speed",
        default=16,
        min=1,
        max=1024,
    )

class VIEW3D_PT_gpu_vertex_paint(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Vertex Paint'
    bl_label = "GPU Light Baker"

    def draw(self, context):
        layout = self.layout
        props = context.scene.adv_vertex_paint_props
        
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(props, "samples")
        box.prop(props, "intensity")
        box.prop(props, "shadow_color")
        
        layout.label(text="Uses Scene Lights (Sun, Point, Area)", icon='LIGHT_DATA')
        
        row = layout.row()
        row.scale_y = 1.5
        row.operator("mesh.gpu_vertex_paint", icon='SHADING_RENDERED')

classes = (
    AdvVertexPaintProperties,
    MESH_OT_gpu_vertex_paint,
    VIEW3D_PT_gpu_vertex_paint,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.adv_vertex_paint_props = bpy.props.PointerProperty(type=AdvVertexPaintProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.adv_vertex_paint_props

if __name__ == "__main__":
    register()