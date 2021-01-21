#pragma once
/// copyright,  qingfeng xia, personal work, not belong to UKAEA
/// Created on Sunday, Nov 01 2020
/// split into occ_project.h, Dec 23 2020
/// Although inscribe just needs 1 view, in order to reuse the code, 3 view are created

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <unordered_map>
#include <algorithm>    // std::sort
#include <stdexcept>
#include <system_error>
#include <optional>

#include "gp_Vec.hxx"
#include "Poly_Triangulation.hxx"

#include "TopoDS.hxx"
#include "TopoDS_Shape.hxx"
#include "TopoDS_Face.hxx"


// these below may be moved into cpp file
#include "Standard_Version.hxx"

#include "gp_Lin.hxx"
#include "Bnd_OBB.hxx"
#include "BRepBndLib.hxx"
#include "BRep_Tool.hxx"
#include "BRepTools.hxx"

#include "BRepMesh_IncrementalMesh.hxx"
#include "Poly_Array1OfTriangle.hxx"
#include "TColgp_Array1OfPnt.hxx"
//#include "TNCOLStd_Array1OfInteger.hxx"
#include "Poly_PolygonOnTriangulation.hxx"

#include "TopTools_HSequenceOfShape.hxx"
#include "TopTools_IndexedMapOfShape.hxx"
#include "TopExp_Explorer.hxx"
#include "TopExp.hxx"

#include "TopLoc_Location.hxx"
#include "BRep_Builder.hxx"
#include "BRepBuilderAPI_Transform.hxx"
#include "BRepBuilderAPI_MakeEdge.hxx"

#include "GProp_GProps.hxx"
#include "BRepGProp.hxx"
#include "BRepAlgoAPI_Common.hxx"

#include "OSD_Path.hxx"
#include "StlAPI_Reader.hxx"
#if OCC_VERSION_HEX >= 0x070300
#include "RWStl.hxx"  // OCCT 7.4, read Shape from STL file
#endif
#include "StlAPI_Writer.hxx"

typedef gp_Vec Coordinate;

#include <Eigen/Dense>
typedef double scalar;  // single-precision float can be push to GPU for acceleration

#ifdef BUILD_INSCRIBE
const unsigned int NVIEWS = 3;
#else
const unsigned int NVIEWS = 3;
#endif
const int DIM=3;

// this can be set, but must be set before call any functions.
std::array<size_t, DIM> NGRIDS = {64, 64, 64};  // make it diff to test image size
const bool NORMALIZED_THICKNESS = true;  // normalized to [0,1] according to boundbox minmax

// normalized XYZ according to OBB, if input geometry has been oriented, then use AABB, not OBB
bool USE_OBB = false;

template <typename T> 
using Mat = Eigen::Matrix<std::shared_ptr<T>, Eigen::Dynamic, Eigen::Dynamic> ;

/// shared_pointer to empty vector instead of nullptr
typedef Mat<std::vector<scalar>> IntersectionMat;
typedef std::array<std::shared_ptr<IntersectionMat>, NVIEWS> IntersectionData;
typedef std::vector<scalar> BoundBoxType;

template <typename T> 
void init_matrix(Mat<T>& mat)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            mat(r, c) = std::make_shared<T>();
        }
    }
}


/// linear point grid (equally-spaced in one axis), 3D grid, slightly bigger than bound box
struct GridInfo
{
    std::array<double, DIM> starts; // 
    std::array<double, DIM> spaces;
    std::array<size_t, DIM> nsteps;  // npixels (cells) in each axis
    std::array<double, DIM> mins; // 
    std::array<double, DIM> maxs;
};

struct MeshData
{
    Handle(Poly_Triangulation) triangles;
    GridInfo grid_info;
    gp_Trsf local_transform;
};

/// normalized to aspect ratios all one
GridInfo generate_grid(const BoundBoxType bbox)
{
    double xmin, ymin, zmin, xmax, ymax, zmax;
    xmin = bbox[0], ymin = bbox[1], zmin = bbox[2];
    xmax = bbox[3], ymax = bbox[4], zmax = bbox[5];

    int m = 1;  //  margin extention on each side, must be integer 1 cell width, other value not tested
    std::array<double, DIM> spaces = {(xmax-xmin)/(NGRIDS[0]-2*m), 
            (ymax-ymin)/(NGRIDS[1]-2*m), (zmax-zmin)/(NGRIDS[2]-2*m)};
    std::array<double, DIM> starts = {xmin - spaces[0] * 0.5 * m, 
            ymin - spaces[1] * 0.5 * m, zmin - spaces[2] * 0.5 * m};  // first cell's center
    // {xmin, ymin, zmin}, {xmax, ymax, zmax}, 
    GridInfo gInfo{starts, spaces, {NGRIDS[0]-1, NGRIDS[1]-1, NGRIDS[1]-1},
        {xmin, ymin, zmin}, {xmax, ymax, zmax}};  // 

    return gInfo;
}

BoundBoxType calcBoundBox(const TopoDS_Shape& shape)
{
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    double xmin, xmax, ymin, ymax, zmin, zmax;
    bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    return {xmin, ymin, zmin, xmax, ymax, zmax};
}

GridInfo generate_grid(const TopoDS_Shape& shape)
{
    // must be rotated with AABB = OBB,  AABB (Axis-align) is `class Bnd_Box`
    return generate_grid(calcBoundBox(shape));
}

/// currently brep only, assuming single solid
TopoDS_Shape read_geometry(std::string filename)
{
    // 2. image resolution, setup grid
    TopoDS_Shape shape;
    BRep_Builder builder;
    BRepTools::Read(shape, filename.c_str(), builder);
    return shape;
}

TopoDS_Shape  rotate_shape(const TopoDS_Shape& shape)
{
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    double xmin, xmax, ymin, ymax, zmin, zmax;
    bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    double xL=xmax-xmin;
    double yL=ymax-ymin;
    double zL=zmax-zmin;

    gp_Trsf trsf;
    if(NORMALIZED_THICKNESS)
    {
        trsf.SetRotation(gp::OX(), M_PI/4.0);    // will rotation add up?
        trsf.SetRotation(gp::OY(), M_PI/4.0);    
        //trsf.SetRotation(gp::OZ(), M_PI/4.0);    
    }
    else
    {
        trsf.SetRotation(gp::OX(), std::atan(zL/yL));    
        trsf.SetRotation(gp::OY(), std::atan(xL/zL));    
        //trsf.SetRotation(gp::OZ(), std::atan(yL/xL)); 
    } 
    auto t = BRepBuilderAPI_Transform(shape, trsf, true);

    return t.Shape();
}

/// for STEP, BREP shape input files, first of all, imprint and merge
/// rotate shape with OBB to AABB, can also translate
TopoDS_Shape  prepare_shape(std::string input, Bnd_OBB& obb)
{
    // 1. rotate view according to boundbox
    TopoDS_Shape shape = read_geometry(input);

    BRepBndLib::AddOBB(shape, obb);
    if(!obb.IsAABox()  &&  USE_OBB)
    {
        gp_Ax3 obb_ax(obb.Center(), obb.ZDirection(), obb.XDirection());
        // occt 7.3 does not have Position() to return gp_Ax3, so create this ax3
        gp_Trsf trsf;
        //gp_Ax3 gc_ax3; // the reference coordinate system (OXYZ)
        //gc_ax3.SetLocation(obb.Center());
        trsf.SetTransformation(obb_ax, gp::XOY());                            
        auto t = BRepBuilderAPI_Transform(shape, trsf, true);

        shape = t.Shape();  // it works
    }
    return shape;
}

/// free (used once in topology), not internal (embedded inside a solid)
std::shared_ptr<std::vector<TopoDS_Face>> get_free_faces(const TopoDS_Shape &shape) 
{
    auto freeFaces = std::make_shared<std::vector<TopoDS_Face>>();

    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        const TopoDS_Face &face = TopoDS::Face(ex.Current());
        // check if it is an external face, skip interior face

        // Important note: For the surface map, face equivalence is defined
        // by TopoDS_Shape::IsSame(), which ignores the orientation.
        // TopoDS_Shape::IsFree()
        //if (face.Orientation() == TopAbs_EXTERNAL)    
            freeFaces->push_back(face);
    }
    return freeFaces;
}

/// gp_Trsf& local_transform is an output parameter
const Handle(Poly_Triangulation)  generate_mesh(TopoDS_Face f, gp_Trsf& local_transform, double linDefl= 0.25)
{
    const double theAngDeflection = 0.5; //default 0.5, degree or radian?
    // mesh each face, by given best tols
    BRepMesh_IncrementalMesh facets(f, linDefl, true, theAngDeflection);

    TopLoc_Location local_loc;
    // BRep_Tool::Triangulation is for face only, return null if no mesh
    const Handle(Poly_Triangulation) triangles  = BRep_Tool::Triangulation(f, local_loc);
    // set output parameter: local transform
    local_transform = local_loc;
    // why TopoLoc_Location  can be assigned to gp_Trsf?

    if (triangles.IsNull()) {
        std::cout << "No facets for surface" << std::endl;
    }

    return triangles;
}

/// OFF STL surface mesh, OCCT 7.4
Handle(Poly_Triangulation)  read_mesh(std::string filename)
{
    #if OCC_VERSION_HEX >= 0x070300
    auto t = RWStl::ReadFile(filename.c_str());
    if(t)
        return t;
    else
        throw std::runtime_error(std::string(" can not find triangle from STL file"));    
    #else
    throw std::runtime_error(std::string("occ 7.4 is needed to read stl mesh as Poly_Triangulation"));
    #endif
}



/// http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
/// https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
bool in_triangle(double xx, double yy, double x[3], double y[3])
{
    double  denominator = ((y[1] - y[2])*(x[0] - x[2]) + (x[2] - x[1])*(y[0] - y[2]));
    double  a = ((y[1] - y[2])*(xx - x[2]) + (x[2]- x[1])*(yy - y[2])) / denominator;
    double  b = ((y[2] - y[0])*(xx - x[2]) + (x[0] - x[2])*(yy - y[2])) / denominator;
    double  c = 1 - a - b;
    
    return 0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1;
}

/// calc of normal depends on triangle vertices seq
double area(const std::vector<const Coordinate*> tri)
{
    gp_Vec ab = (*tri[1]) - (*tri[0]);
    gp_Vec ac = (*tri[2]) - (*tri[0]);
    gp_Vec n = ab.Crossed(ac);      // cross product
    // divided by 2 is the triangle area
    return n.Magnitude() /2.0;
}

/// Möller and Trumbore, « Fast, Minimum Storage Ray-Triangle Intersection », Journal of Graphics Tools, vol. 2,‎ 1997, p. 21–28
/// it is more efficient on GPU
/// http://geomalgorithms.com/a06-_intersect-2.html
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 =  disjoint (no intersect)
//             1 =  intersect in unique point I1
//             2 =  are in the same plane
int calc_intersection(const std::vector<const Coordinate*> tri, const std::vector<const Coordinate*> R, gp_Vec& I)
{
    gp_Vec    u, v, n;              // triangle vectors
    gp_Vec    w0, w;                // ray vectors
    scalar     r, a, b;              // params to calc ray-plane intersect
    const scalar SMALL_NUM = 1e-30;    //

    // get triangle edge vectors and plane normal
    u = (*tri[1]) - (*tri[0]);
    v = (*tri[2]) - (*tri[0]);
    n = u.Crossed(v);                 // cross product
    if (n.Magnitude() < 1e-308)    // triangle is degenerate, zero area?
        return -1;                     // do not deal with this case
         
    w0 = *(R[0]) - (*tri[0]);
    gp_Vec dir = *(R[1]) - *(R[0]);
    a = -n.Dot(w0);
    b = n.Dot(dir);           // ray direction vector
    if (std::fabs(b) < SMALL_NUM) {     // ray is  parallel to triangle plane
        if (a == 0)                 // ray lies in triangle plane, co-planar
            return 2;
        else 
            return 0;              // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)                    // ray goes away from triangle
        return 0;                   // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    I = *R[0] + r * dir;            // intersect point of ray and plane

    scalar  uu, uv, vv, wu, wv, D;
    uu = u.Dot(u);
    uv = u.Dot(v);
    vv = v.Dot(v);
    w = I - (*tri[0]);
    wu = w.Dot(u);
    wv = w.Dot(v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    scalar s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)         // I is outside T
        return 0;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return 0;

    return 1;
}

bool has_intersection(const std::vector<const Coordinate*> poly, double vi, double vj, int iaxis)
{
    double x[3], y[3];
    for(int i = 0; i < 3; i++)
    {
        x[i] = (*poly[i]).Coord(iaxis+1);
        int iy = iaxis + 2 > 3? 1:iaxis + 2;
        y[i] = (*poly[i]).Coord(iy);
    }
    
    return in_triangle(vi, vj, x, y);
}


void init_intersection_data(IntersectionData& data)
{

    data[0] = std::make_shared<IntersectionMat>(NGRIDS[0], NGRIDS[1]);
    data[1] = std::make_shared<IntersectionMat>(NGRIDS[1], NGRIDS[2]);
    data[2] = std::make_shared<IntersectionMat>(NGRIDS[2], NGRIDS[0]);

    for(int i=0; i<NVIEWS; i++)
    {
        IntersectionMat& mat = (*data[i]);
        init_matrix(mat);
    }
}

void sort_intersection(IntersectionMat& mat)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            auto& p = mat(r,c);
            if (p and p->size())
                std::sort((*p).begin(), (*p).end());           
        }
    }
}

void normalize_intersection(IntersectionMat& mat, const std::pair<scalar, scalar> minmax)
{
    auto fh = minmax.second - minmax.first;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            auto& p = mat(r,c);
            if (p and p->size())
            {
                for(size_t i=0; i<p->size(); i++)
                    (*p)[i] = ((*p)[i] - minmax.first) / fh;
            }         
        }
    }
}


std::vector<std::array<size_t,2>> get_index_ranges(const std::vector<const Coordinate*>& polygon, const GridInfo gInfo)
{
    std::vector<std::array<size_t,2>> ranges;
    for(int iaxis = 0; iaxis < DIM; iaxis++)
    {
        std::vector<double> v;
        for(const auto& p: polygon)
        {
            v.push_back((*p).Coord(iaxis+1));   /// NOTE: index for X is 1 not 0
        }
        auto min_i = *std::min_element(v.cbegin(), v.cend());
        // what happen if the mesh vertex coincides with projection mesh, minus 1 to be conservative
        int imin = int((min_i - gInfo.starts[iaxis]) / gInfo.spaces[iaxis] ) - 1;
        if(imin < 0)  // comparison with zero will be always true,   size_t i= 0,  i-1  is a very big number
        {
            imin = 0;
        }
        auto max_i = *std::max_element(v.cbegin(), v.cend());
        /// should be rounded up for max, not to miss some point
        size_t imax = size_t((max_i - gInfo.starts[iaxis]) / gInfo.spaces[iaxis] ) + 1;
        if (imax > gInfo.nsteps[iaxis])
        {
            std::cout << "Error:  i_upper " << imax << " >= grid number " << gInfo.nsteps[iaxis] << std::endl;
            imax = gInfo.nsteps[iaxis];
        }
        ranges.push_back({imin, imax});
    }
    return ranges;
}

//const std::vector<Coordinate>& points, const std::array<int, 3>& tri
void insert_intersections(const std::vector<const Coordinate*>& triangle, const GridInfo& gInfo, IntersectionData& data)
{
    // get box bound for the tri, project to one plane, get possible indices from the grid
    const auto iranges = get_index_ranges(triangle, gInfo);  /// BUG solved, negative int becomes very big size_t number
    //std::vector<std::array<size_t,2>> iranges = {{0, NGRIDS[0]}, {0, NGRIDS[1]}, {0, NGRIDS[2]}};

    // if line and triangle intersection, then save the intersection coord's third-component
    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        const size_t r_start = iranges[r_index][0];
        const size_t r_end = iranges[r_index][1];
        size_t c_index = (1 + iaxis)==3? 0: 1 + iaxis;
        const size_t c_start = iranges[c_index][0];
        const size_t c_end = iranges[c_index][1];
        size_t third_index = (iaxis + 2)%3;
        const size_t t_start = iranges[third_index][0];
        const size_t t_end = iranges[third_index][1];
        gp_Vec dir(0, 0, 0);
        dir.SetCoord(third_index + 1, 1.0);
        // this ray must be bigger than boundbox range (min, max), it is fine here
        double third_min = gInfo.starts[third_index] - gInfo.spaces[third_index] * gInfo.nsteps[third_index];
        double third_max = gInfo.starts[third_index] + gInfo.spaces[third_index] * (gInfo.nsteps[third_index]*2);

        double h_min = gInfo.starts[third_index] - gInfo.spaces[third_index] * t_start;
        double h_max = gInfo.starts[third_index] + gInfo.spaces[third_index] * t_end;

        IntersectionMat& mat = *data[iaxis];
        for(size_t r =  r_start; r <= r_end; r++)  // r_end must also been used, closed range
        {
            for(size_t c = c_start; c <= c_end; c++)
            {
                double r_value = gInfo.starts[r_index] + gInfo.spaces[r_index] * r;
                double c_value = gInfo.starts[c_index] + gInfo.spaces[c_index] * c;

                gp_Vec pos(0, 0, 0);  // occt index starts at 1, not zero
                pos.SetCoord(r_index+1, r_value);
                pos.SetCoord(c_index+1, c_value);
                pos.SetCoord(third_index + 1, third_min);
                gp_Vec p2 = pos;
                p2.SetCoord(third_index + 1, third_max);   // evil of copy-and-paste!
                //gp_Lin l(pos, dir);

                #if 0
                auto ret = has_intersection(triangle, r_value, c_value, iaxis);
                if(ret)
                {
                    gp_Vec I(0, 0, 0);
                    int m = calc_intersection(triangle,  {&pos, &p2}, I);
                    //if (m == 1)
                    mat(r, c)->push_back(I.Coord(third_index + 1));
                }
                #else
                gp_Vec I(0, 0, 0);
                int m = calc_intersection(triangle, {&pos, &p2}, I);
                if (m == 1)
                {
                    auto  h = I.Coord(third_index + 1);
                    //if (h < third_max && h > third_min)  // this may needs to be narrowed down.
                    if (h < h_max && h > h_min)
                    {
                        // if (r < r_start)
                        //     std::cout << " intersection found at r index = " << r << " less than r_start =" << r_start << std::endl;
                        // if (c < c_start)
                        //     std::cout << " intersection found at c index = " << c << " less than c_start = " << c_start << std::endl;
                        mat(r, c)->push_back(h);
                    }
                    else
                    {
                        std::cout << " Error: intersection point out of triangle bound box" << std::endl;
                    }
                    
                }
                #endif
            }
        }
    }
}



void calc_intersections(const Handle(Poly_Triangulation) triangles, const GridInfo& ginfo, const gp_Trsf local_transform, IntersectionData& data) 
{
    // mixing of array index starts at 1 and index starts 0 is buggy
    const TColgp_Array1OfPnt &nodes = triangles->Nodes();
    // retrieve triangle coordinates
    std::vector<Coordinate> points;   // NCollection_Array1<>
    points.reserve(nodes.Upper());
    if (nodes.Lower() == 1 )  // index starts at 1
    {
        //points.push_back({INFINITY, INFINITY, INFINITY});  // if we use index 0, then all inf
    }
    for (int i = nodes.Lower(); i <= nodes.Upper(); i++) {
        Standard_Real x, y, z;
        nodes(i).Coord(x, y, z);
        local_transform.Transforms(x, y, z);  // inplace modify
        points.push_back({x,y,z});
    }

    std::array<int, 3> indices;
    const Poly_Array1OfTriangle &tris = triangles->Triangles();
    //     std::cout << "Face has " << tris.Length() << " triangles" << std::endl;
    for (int i = tris.Lower(); i <= tris.Upper(); i++) 
    {
        // get the node indexes for this triangle
        const Poly_Triangle &tri = tris(i);
        tri.Get(indices[0], indices[1], indices[2]);
        std::vector<const Coordinate*> triangle = {points.data() + indices[0]-1,  
                points.data() + indices[1]-1,  points.data() + indices[2]-1};
        insert_intersections(triangle, ginfo, data);
    }
}

