
#include <json.hpp>
#include <GProp_GProps.hxx>
#include <BRepGProp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Shape.hxx>
#include <Bnd_OBB.hxx>
#include <BRepBndLib.hxx>
//#include <TopoDS_Shape.hxx>


using json = nlohmann::json;

// adapted from ppp/Geom/PropertyBuilder
void writeMetadataFile(const TopoDS_Shape& shape, const std::string file_name)
{
    std::ofstream o(file_name);

    GProp_GProps v_props, s_props, l_props;
    BRepGProp::LinearProperties(shape, l_props);
    BRepGProp::SurfaceProperties(shape, s_props);
    BRepGProp::VolumeProperties(shape, v_props);

    Bnd_OBB obb;
    BRepBndLib::AddOBB(shape, obb);
    
    int edgeCount = 0;
    TopExp_Explorer Ex(shape, TopAbs_EDGE);
    while (Ex.More())
    {
        edgeCount++;
        Ex.Next();
    }

    int faceCount = 0;
    {
        TopExp_Explorer Ex(shape, TopAbs_FACE);
        while (Ex.More())
        {
            faceCount++;
            Ex.Next();
        }
    }

    int solidCount = 0;
    {
        TopExp_Explorer Ex(shape, TopAbs_SOLID);
        while (Ex.More())
        {
            solidCount++;
            Ex.Next();
        }
    }

    auto center = v_props.CentreOfMass();
    std::vector<double> centerOfMass;
    centerOfMass.push_back(center.X());
    centerOfMass.push_back(center.Y());
    centerOfMass.push_back(center.Z());

    std::vector<double> obbSizes;
    obbSizes.push_back(obb.XHSize());
    obbSizes.push_back(obb.YHSize());
    obbSizes.push_back(obb.ZHSize());

    json j{{"volume", v_props.Mass()},
            {"area", s_props.Mass()},
            {"perimeter", l_props.Mass()},
            {"edgeCount", edgeCount},
            {"faceCount", faceCount},
            {"solidCount", solidCount},
            {"center", centerOfMass},
            {"obb", obbSizes}
    };
    // todo: obb boundbox  to json

    o << std::setw(4);
    o << j;
    o << std::endl;
}