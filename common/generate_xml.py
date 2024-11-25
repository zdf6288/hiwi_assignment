import xml.etree.ElementTree as ET
import numpy as np

def generate_scene_xml(start, end, obstacles, output_file="scene_with_obstacle_generated.xml"):
    """
    Generate a MuJoCo XML file based on start position, end position, and obstacles.

    :param start: Start position (x, y, z), array or list
    :param end: End position (x, y, z), array or list
    :param obstacles: List of obstacles, each defined as {"position": [x, y, z], "radius": r}
    :param output_file: Path to save the generated XML file
    """
    # Create the root element <mujoco>
    mujoco = ET.Element("mujoco", {"model": "dynamic_obstacle_scene"})
    compiler = ET.SubElement(mujoco, "compiler", {"angle": "radian"})
    option = ET.SubElement(mujoco, "option", {"timestep": "0.01"})

    # <worldbody> element
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Ground plane
    ET.SubElement(worldbody, "geom", {
        "name": "ground",
        "type": "plane",
        "pos": "0 0 0",
        "size": "2 2 0.1",
        "rgba": "0.9 0.9 0.9 1"
    })

    # Start point marker
    ET.SubElement(worldbody, "site", {
        "name": "start",
        "pos": f"{start[0]} {start[1]} {start[2]}",
        "size": "0.02",
        "rgba": "0 1 0 1"
    })

    # End point marker
    ET.SubElement(worldbody, "site", {
        "name": "end",
        "pos": f"{end[0]} {end[1]} {end[2]}",
        "size": "0.02",
        "rgba": "0 0 1 1"
    })

    # Add obstacles
    for i, obstacle in enumerate(obstacles):
        ET.SubElement(worldbody, "geom", {
            "name": f"obstacle_{i}",
            "type": "sphere",
            "pos": f"{obstacle['position'][0]} {obstacle['position'][1]} {obstacle['position'][2]}",
            "size": f"{obstacle['radius']}",
            "rgba": "1 0 0 1"
        })

    # Dynamic trajectory marker
    ET.SubElement(worldbody, "body", {"name": "trajectory_marker", "mocap": "true"}).append(
        ET.Element("geom", {"type": "sphere", "size": "0.02", "rgba": "1 1 0 1"})
    )

    # Save the XML file
    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="\t", level=0)  # Beautify indentation
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"XML file generated: {output_file}")
    
    return output_file