/// copyright,  qingfeng xia, personal work, not belong to UKAEA
/// Created on Sunday, Nov 01 2020

#include "occ_project.h"
#include "geometryProperty.cpp"

typedef Eigen::MatrixXf Image;
const std::vector<std::string> ORTH_NAMES = {"_XY", "_YZ", "_ZX"};  // _XY is project from Z-axis, orth
const std::vector<std::string> TRI_NAMES = {"_TRI_XY", "_TRI_YZ", "_TRI_ZX"}; 
const std::vector<std::string> RZ_NAMES = {"_RZ_XY", "_RZ_YZ", "_RZ_ZX"}; 
const std::vector<std::string> RY_NAMES = {"_RY_XY", "_RY_YZ", "_RY_ZX"}; 
const std::vector<std::string> RX_NAMES = {"_RX_XY", "_RX_YZ", "_RX_ZX"};
/// corresponding to projection_id, X-axis = 0, 
const std::vector<const std::vector<std::string>*> PROJ_NAMES = {
    &RX_NAMES, &RY_NAMES, &RZ_NAMES, &TRI_NAMES, &ORTH_NAMES
};
const int TRI_PROJECTION = 3;
const int NO_ROTATION = 4;
const bool MERGE_3_INTERSECTION = true;  // ModeNet stl mesh has self-intersection faces

//bool USE_ISOMETRIC = false;   // currently
// bool ROTATE_ZAXIS = false;
// bool ROTATE_YAXIS = false;
// bool ROTATE_XAXIS = false;

#define DUMP_BOP_INTERSECTED_EDGES 0
#define DUMP_GRID 0
#define DUMP_PART_MESH 0
#define USE_OPENCV 1

const double error_threshold = 0.03;  // less than this ratio will be interpolated

#if USE_OPENCV
// OpenImageIO can be another choice
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
const std::string IM_SUFFIX = ".png";
#else
const std::string IM_SUFFIX = ".csv";
#endif



// PCL has a tool `mesh2pcd`: convert a CAD model to a PCD (Point Cloud Data) file, using ray tracing operations.

gp_Trsf generate_transformation(BoundBoxType bbox,  int projection_id = 5)
{
    double xL = bbox[3]-bbox[0];
    double yL = bbox[4]-bbox[1];
    double zL = bbox[5]-bbox[2];

    const gp_Ax1 axes[] = {gp::OX(), gp::OY(), gp::OZ()};
    gp_Trsf trsf;

    if (projection_id == NO_ROTATION)
        return trsf;

    if (projection_id == TRI_PROJECTION)
    {
       throw std::runtime_error("not impl"); 
    }
    int axis_id = projection_id;
    if( USE_CUBE_BOX ) 
    {
        trsf.SetRotation(axes[axis_id], M_PI/4.0);    // will rotation add up? NO
    }
    else
    {
        trsf.SetRotation(axes[axis_id], M_PI/4.0);    // will rotation add up? NO
        //trsf.SetRotation(axes[axis_id], std::atan(zL/yL));
        // throw std::runtime_error("not impl");
    }
    return trsf;
}

/// axis_id == 2 for rotating around Z-axis, axis_id = 0 for x-axis
TopoDS_Shape  transform_shape(const TopoDS_Shape& shape, int projection_id)
{
    // boundbox is only needed if USE_CUBE_BOX
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);

    gp_Trsf trsf = generate_transformation(toBoundBox(bbox), projection_id);
    auto t = BRepBuilderAPI_Transform(shape, trsf, true);

    return t.Shape();
}

//typedef std::array<double, 3> Coordinate;   gp_Pnt,  gp_XYZ instead
void save_image(const Image& im, const std::string& filename)
{
    const static Eigen::IOFormat CSVFormat(4, 0, ", ", "\n");
    std::ofstream file(filename.c_str());
    file << im.format(CSVFormat);
}

/// Image has been sorted and normalized into [0, 1] range, then save to png file
void save_image(const Image& tim, const std::string& filename, const IntersectionMat& mat)
{
    cv::Mat im(mat.rows(), mat.cols(), CV_8UC3, cv::Scalar(0,0,0));
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();
            if(n>=2)
            {
                im.at<cv::Vec3b>(r, c) = {(*p)[0] *255, tim(r,c) * 255, (*p)[n-1] *255}; // will not overflow
            }
        }
    }
    cv::imwrite(filename, im);
}

void calc_nearest(const IntersectionMat& mat, Image& im)
{
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();
            if(n==0)
                im(r, c) = 0.0f;
            else
                im(r, c) = (*p)[0]; // assuming has been sorted
        }
    }
}

/// filter out values with similar values, input vector must be sorted
std::vector<scalar> remove_duplicated_values(const std::vector<scalar>& hv)
{
    std::vector<scalar> r;
    const scalar precision = 1e-1;

    r.push_back(hv[0]);
    for(size_t i=0; i<hv.size()-1; i++)
    {
        if (std::abs(hv[i+1] - hv[i]) > precision)
            r.push_back(hv[i+i]);
    }
    return r;
}

/// input vector must be sorted, and vector size is even number
scalar calc_thickness(const std::vector<scalar>& v)
{
    double t = 0.0;
    for(size_t i = 0; i<v.size()/2; i++)
    {
        t +=  v[i*2+1] - v[i*2];
        if (v[i*2+1] - v[i*2] < 0)  // should not happen if all sorted
        {
            std::cout << "Error: intersection height vector is not sorted!\n";
        }
    }
    return t;
}

std::set<std::pair<size_t, size_t>> calc_thickness(const IntersectionMat& mat, Image& im)
{
    std::set<std::pair<size_t, size_t>> error_pos;
    for (size_t r=0; r< mat.rows(); r++)
    {
        for (size_t c=0; c< mat.cols(); c++)
        {
            const auto& p = mat(r,c);
            const auto n = p->size();
            if(n==0)
                im(r, c) = 0.0f;
            else if (n%2 == 1)
            {
                if(MERGE_3_INTERSECTION and n==3)
                {
                    auto min = *std::min_element(p->begin(), p->end());
                    auto max = *std::max_element(p->begin(), p->end());
                    im(r, c) = std::abs(max - min); 
                }
                else{
                    std::cout << "Error: intersection vector size " << n << " is not an even number " << r << ", " << c <<"\n";
                    auto fhv = remove_duplicated_values(*p);

                    if (fhv.size() % 2 == 0)
                    {
                        std::cout << "Because there is duplicated height values, filtered out and calc thickness \n";
                        im(r, c) = calc_thickness(fhv);
                    }
                    else{
                        im(r, c) = 0.0f;   // set zero, or intoperation later
                        error_pos.insert({r, c});
                    }
                }
            }
            else if(n==2)
                im(r, c) = std::abs((*p)[0] - (*p)[1]);
            else
            {
                im(r, c) = calc_thickness(*p);
            }
        }
    }
    return error_pos;
}

/// if single pixel missing, it is acceptable for interpolation
/// if there is large area of missing pixels, interpolation has bad quality.
void interoplate_thickness(const std::set<std::pair<size_t, size_t>> error_pos, Image& im)
{
    for(const auto& it: error_pos)
    {
        size_t r = it.first;
        size_t c = it.second;
        std::vector<float> values;
        for (size_t i=1; r+i < im.rows(); i++)
        {
            if( error_pos.find({r+i, c}) == error_pos.end() )
            {
                values.push_back(im(r+i, c));
                break; 
            }
        }
        for (size_t i=1; r-i >= 0; i++)
        {
            if( error_pos.find({r-i, c}) == error_pos.end())
            {
                values.push_back(im(r-i, c));
                break; 
            }
        }
        for (size_t i=1; c+i < im.cols(); i++)
        {
            if( error_pos.find({r, c+i}) == error_pos.end() )
            {
                values.push_back(im(r, c+i));
                break; 
            }
        }
        for (size_t i=1; c-i >= 0; i++)
        {
            if( error_pos.find({r, c-i}) == error_pos.end() )
            {
                values.push_back(im(r, c-i));
                break; 
            }
        }
        assert(std::size(values) <= 4);
        im(r, c) = std::accumulate(values.cbegin(), values.cend(), 0.0) / std::size(values);
    }
}


void test_IndexedMap()
{
    // is key in set ordered? will insertion change index or previous?
    NCollection_IndexedDataMap<const char*, int> map;
    auto i1 = map.Add("B", 1);
    auto i2 = map.Add("A", 2);
    std::cout << "NCollection_IndexedDataMap index: " << i1 << i2 << std::endl;
}

/// code extracted from `calc_thickness_bop`
void dump_grid(const GridInfo& gInfo, std::string input)
{
    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        size_t c_index = (1 + iaxis)%3;
        size_t z_index = (iaxis + 2)%3;
        //gp_Vec dir(0, 0, 0);
        //dir.SetCoord(z_index + 1, 1.0);

        TopoDS_Builder cBuilder;
        TopoDS_Compound merged;
        cBuilder.MakeCompound(merged);
        scalar zStart = gInfo.starts[z_index] + gInfo.spaces[z_index] * -1;
        scalar zEnd = gInfo.starts[z_index] + gInfo.spaces[z_index] * ( gInfo.nsteps[z_index] + 1);

        for (size_t r=0; r< gInfo.nsteps[r_index] + 1; r++)
        {
            double r_value = gInfo.starts[r_index] + gInfo.spaces[r_index] * r;
            for (size_t c=0; c< gInfo.nsteps[c_index] + 1; c++)
            {
                double c_value = gInfo.starts[c_index] + gInfo.spaces[c_index] * c;
                // create the curve, then surface, IntCS
                gp_Pnt p1; 
                p1.SetCoord(r_index+1, r_value);
                p1.SetCoord(c_index+1, c_value); 
                p1.SetCoord(z_index+1, zStart);
                gp_Pnt p2 = p1;
                p2.SetCoord(z_index+1, zEnd);

                auto e = BRepBuilderAPI_MakeEdge(p1, p2).Edge();

                cBuilder.Add(merged, e);
            }
        }

        //cBuilder.Add(merged, shape);
        BRepTools::Write(merged, (input + ORTH_NAMES[iaxis] + "_grid.brep").c_str());
    }
}

/// find after faceting, 
void save_data(IntersectionData& data, const std::string& output_file_stem, 
                const GridInfo& gInfo, const std::vector<std::string> PNAMES)
{

    for(int i=0; i<NVIEWS; i++)
    {
        auto& mat = *data[i];
        size_t thickness_i = (i+2)%DIM;
        std::pair<scalar, scalar> minmax = {gInfo.mins[thickness_i], gInfo.maxs[thickness_i]};
        sort_intersection(mat);
        if(NORMALIZED_THICKNESS)
            normalize_intersection(mat, minmax);

        Image im(mat.rows(), mat.cols());
        auto error_pos = calc_thickness(mat, im);

        if (std::size(error_pos) > mat.rows() * mat.cols() * error_threshold)
        {
            std::cout << "Error: image will not be saved as total number of odd thickness vector size " << 
                std::size(error_pos) << " is higher than the threshold percentage " <<  error_threshold * 100 <<"\n";
            continue;
        }
        else
        {
            if(std::size(error_pos) > 0)
            {
                std::cout << "Warning: total number of odd thickness vector size " << std::size(error_pos) << 
                    ",  less than threshold " <<  error_threshold * 100 << " percentage, will be interpolated for " << PNAMES[i] << "\n";
                interoplate_thickness(error_pos, im);
            }
            #if USE_OPENCV
            save_image(im, output_file_stem + PNAMES[i] + IM_SUFFIX, mat);
            #else
            save_image(im, output_file_stem + PNAMES[i] + IM_SUFFIX);
            if (NORMALIZED_THICKNESS)
            {
                Image sim(mat.rows(), mat.cols());
                calc_nearest(mat, sim);
                save_image(im, output_file_stem + "_nearest"+ PNAMES[i] + IM_SUFFIX);
            }
            #endif
        }
    }
#if DUMP_GRID
    dump_grid(gInfo, output_file_stem);
#endif
}


/// slow but robust: find the intersection line segments by boolean operation
double intersect_thickness_bop(const TopoDS_Shape& shape, const TopoDS_Edge& edge)
{
    double t = 0;

    BRepAlgoAPI_Common bc{shape, edge};
    //bc.SetRunParallel(occInternalParallel);
    // bc.SetFuzzyValue(); tolerance?
    bc.SetNonDestructive(Standard_True);
    if (not bc.IsDone())
    {
        std::cout << "BRepAlgoAPI_Common is not done\n";
    }

    std::vector<TopoDS_Shape> res; // output is limited to Solid shape type
    for (TopExp_Explorer anExp(bc.Shape(), TopAbs_EDGE); anExp.More(); anExp.Next())
    {
        GProp_GProps l_props;
        BRepGProp::LinearProperties(anExp.Current(), l_props, true);
        t +=  l_props.Mass();
    }

    return t;
}

#if 0
/// this BOP method is very slow but precise
///  and it uses a grid different from triangulation method
int bop(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    double min[DIM], max[DIM], space[DIM];
    box.Get(min[0], min[1], min[2], max[0], max[1], max[2]);

    for (size_t i=0; i< DIM; i++)
    {
        #if 1  // to give margin
        space[i] = (max[i] - min[i])/(NGRIDS[i]-2);
        min[i] -= space[i];
        #else
        space[i] = (max[i] - min[i])/(NGRIDS[i]);
        #endif
    }

    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        size_t r_index = iaxis;
        size_t c_index = (1 + iaxis)%3;
        size_t z_index = (iaxis + 2)%3;
        //gp_Vec dir(0, 0, 0);
        //dir.SetCoord(z_index + 1, 1.0);

        Image im(NGRIDS[r_index], NGRIDS[c_index]);

        TopoDS_Builder cBuilder;
        TopoDS_Compound merged;
        cBuilder.MakeCompound(merged);

        for (size_t r=0; r< im.rows(); r++)
        {
            double r_value = min[r_index] + space[r_index] * (r + 0.5);
            for (size_t c=0; c< im.cols(); c++)
            {
                double c_value = min[c_index] + space[c_index] * (c + 0.5);
                // create the curve, then surface, IntCS
                gp_Pnt p1; 
                p1.SetCoord(r_index+1, r_value);
                p1.SetCoord(c_index+1, c_value); 
                p1.SetCoord(z_index+1, min[z_index]);
                gp_Pnt p2 = p1;
                p2.SetCoord(z_index+1, max[z_index]);

                auto e = BRepBuilderAPI_MakeEdge(p1, p2).Edge();
                im(r, c) = intersect_thickness_bop(shape, e);
                cBuilder.Add(merged, e);
            }
        }

        save_image(im, output_file_stem + "_BOP" + PNAMES[iaxis] + IM_SUFFIX);
#if DUMP_GRID
        cBuilder.Add(merged, shape);
        BRepTools::Write(merged, (output_file_stem + PNAMES[iaxis] + "_grid.brep").c_str());
#endif
    }
    // BRepTools::Write(shape, (output_file_stem + "_shape.brep").c_str());
    return 0;
}

#endif

/// slow but robust: find the intersection line segments by boolean operation
std::vector<scalar> intersect_heights_bop(const TopoDS_Shape& shape, const TopoDS_Edge& edge, int iCoord, TopoDS_Shape& eIntersected)
{
    BRepAlgoAPI_Common bc{shape, edge};
    //bc.SetRunParallel(occInternalParallel);
    // bc.SetFuzzyValue(); tolerance?
    bc.SetNonDestructive(Standard_True);
    if (not bc.IsDone())
    {
        std::cout << "BRepAlgoAPI_Common is not done\n";
    }

    std::vector<scalar> res; // output is limited to Solid shape type
    eIntersected = bc.Shape();
    for (TopExp_Explorer anExp(eIntersected, TopAbs_EDGE); anExp.More(); anExp.Next())
    {
        const auto ed = anExp.Current();
        std::vector<scalar> h; 

        // ed.Oriented(TopAbs_FORWARD)
        for(TopExp_Explorer exEdge(ed, TopAbs_VERTEX); exEdge.More(); exEdge.Next())
        {
            auto curV = TopoDS::Vertex(exEdge.Current());
            auto p = BRep_Tool::Pnt(curV);
            h.push_back(p.Coord(iCoord));
        }
        assert(h.size() > 0 && h.size() == 2);
        if (h.size() == 1)
        {
            std::cout << "the edge has one vertex, error?: \n";
            // for (auto v: res)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else
        {
            res.push_back(*std::min_element(h.cbegin(), h.cend()));
            res.push_back(*std::max_element(h.cbegin(), h.cend()));
            if (h.size() % 2 == 1  || h.size() > 2)
            {
                std::cout << "the edge has more than 2 vertex \n";
            }
        }
    }

    return res;
}

/// grid has been valid by show grid lines and the shape
void calc_thickness_bop(const TopoDS_Shape& shape, const GridInfo& gInfo, IntersectionData& data)
{
    /*
    double min[DIM], max[DIM], space[DIM];
    for (size_t i=0; i< DIM; i++)
    {
        #if 1  // to give margin
        space[i] = (max[i] - min[i])/(NGRIDS[i]-2);
        min[i] -= space[i];
        #else
        space[i] = (max[i] - min[i])/(NGRIDS[i]);
        #endif
    }
    */

    for(int iaxis = 0; iaxis < 3; iaxis++)
    {
        IntersectionMat& mat = *data[iaxis];
        size_t r_index = iaxis;
        size_t c_index = (1 + iaxis)%3;
        size_t z_index = (iaxis + 2)%3;
        //gp_Vec dir(0, 0, 0);
        //dir.SetCoord(z_index + 1, 1.0);
#if DUMP_BOP_INTERSECTED_EDGES
        TopoDS_Builder cBuilder1;
        TopoDS_Compound merged1;
        cBuilder1.MakeCompound(merged1);
#endif
        TopoDS_Builder cBuilder;
        TopoDS_Compound merged;
        cBuilder.MakeCompound(merged);
        scalar zStart = gInfo.starts[z_index] + gInfo.spaces[z_index] * -1;
        scalar zEnd = gInfo.starts[z_index] + gInfo.spaces[z_index] * ( gInfo.nsteps[z_index] + 1);

        for (size_t r=0; r< mat.rows(); r++)
        {
            double r_value = gInfo.starts[r_index] + gInfo.spaces[r_index] * r;
            for (size_t c=0; c< mat.cols(); c++)
            {
                double c_value = gInfo.starts[c_index] + gInfo.spaces[c_index] * c;
                // create the curve, then surface, IntCS
                gp_Pnt p1; 
                p1.SetCoord(r_index+1, r_value);
                p1.SetCoord(c_index+1, c_value); 
                p1.SetCoord(z_index+1, zStart);
                gp_Pnt p2 = p1;
                p2.SetCoord(z_index+1, zEnd);

                auto e = BRepBuilderAPI_MakeEdge(p1, p2).Edge();
                TopoDS_Shape e1;
                auto v = intersect_heights_bop(shape, e, z_index+1, e1);
                mat(r, c) = std::make_shared<std::vector<scalar>>(std::move(v));
                cBuilder.Add(merged, e);

                #if DUMP_BOP_INTERSECTED_EDGES
                cBuilder1.Add(merged1, e1);
                #endif
            }
        }

        #if DUMP_BOP_INTERSECTED_EDGES
        BRepTools::Write(merged1, ("debug_bop_intersected_" + ORTH_NAMES[iaxis] + "_edges.brep").c_str());
        #endif
#if DUMP_GRID
        cBuilder.Add(merged, shape);
        BRepTools::Write(merged, ("debug_bop" + ORTH_NAMES[iaxis] + "_grid.brep").c_str());
#endif
    }
}



/// triangulation, fast less precise than BOP
int project_geom(std::string input, const std::string& output_file_stem, int projection_id, bool bop = false)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    if (projection_id != NO_ROTATION)
    {
        shape = transform_shape(shape, projection_id);
    }
    MeshData mesh;
    mesh.grid_info = generate_grid(shape);

    IntersectionData data;
    init_intersection_data(data);

    if (bop)
    {
        calc_thickness_bop(shape, mesh.grid_info, data);
    }
    else{
        auto faces = get_free_faces(shape);
        double linDefl = 0.25; // or 0.1 times of pixel grid space?
        for(const auto& f: *faces)
        {
            mesh.triangles = generate_mesh(f, mesh.local_transform, linDefl);
            calc_intersections(mesh.triangles, mesh.grid_info, &mesh.local_transform, data);
        }
    }

    // save thickness matrix as numpy array,  scale it?  save as image?
    if (projection_id == NO_ROTATION)
    {
        save_data(data, output_file_stem, mesh.grid_info, ORTH_NAMES);
        writeMetadataFile(shape, output_file_stem + "_metadata.json");
    }
    else
    {
        save_data(data, output_file_stem, mesh.grid_info, *PROJ_NAMES[projection_id]);
    }


#if 0 // DUMP_PART_MESH
    auto stl_exporter = StlAPI_Writer(); // high level API
    stl_exporter.Write(shape, (input + "_meshed.stl").c_str()); 
    // shape must has mesh for each face
#endif

    return 0;
}

/// once bound box can be calc from mesh, then FreeCAD preprocessor is not needed
int project_mesh(std::string input, const std::string& output_file_stem, int projection_id, const BoundBoxType& bbox)
{
    IntersectionData data;
    init_intersection_data(data);
    auto input_mesh = read_mesh(input);
    // calc bound box from input mesh triangulation
    auto rotated_mesh = transform_triangulation(input_mesh, generate_transformation(bbox, projection_id));

    MeshData mesh;
    mesh.local_transform = gp_Trsf();  // dummy, do nothing
    mesh.triangles = rotated_mesh;
    mesh.grid_info = generate_grid(calc_mesh_boundbox(mesh.triangles));
    // 
    calc_intersections(mesh.triangles, mesh.grid_info, nullptr, data);

    // save thickness matrix as numpy array,  scale it?  save as image?
    save_data(data, output_file_stem, mesh.grid_info, *PROJ_NAMES[projection_id]);

#if DUMP_PART_MESH
    write_mesh(mesh.triangles, input + "_transformed.stl");
#endif
    return 0;
}


#include <argparse.hpp>

int get_projection_id(const argparse::ArgumentParser& program)
{
    if (program.get<bool>("--triview"))
        return 3;
    else if  (program.get<bool>("--rz"))
        return 2;
    else
        return NO_ROTATION;
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("program name");

  program.add_argument("input")
    .required()
    .help("input geometry file path");

  program.add_argument("-o", "--output")
    .help("specify the output file stem");

  program.add_argument("--grid").nargs(3)
    .help("image pixel for x, y, z as an integer array")
    .default_value(std::vector<size_t>{64, 64, 64})
    .action([](const std::string& value) { return std::stoul(value); });

  program.add_argument("--bbox").nargs(6)
    .help("boundbox values for stl off input file: xmin, ymin, zmin, xmax, ymax, zmax")
    .action([](const std::string& value) { return std::stod(value); });

  program.add_argument("--bop")
    .help("use the very slow BOP method")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--triview")
    .help("generate extra tri views")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--rz")
    .help("generate views after rotating Z by 45deg")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--obb")
    .help("use the OBB, instead of default axis-align bound box")
    .default_value(false)
    .implicit_value(true);

  program.add_argument("--cube")
    .help("use the cube bound box (equal max length for xyz), instead of obb or normal box bound box")
    .default_value(false)
    .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }

    test_calc_intersection(true);
    test_calc_intersection(false);

    USE_OBB = program.get<bool>("--obb");  // set global variable
    USE_CUBE_BOX = program.get<bool>("--cube");
    //if(program.present("--grid"))  // present() must not have a default value
    {
        std::vector<size_t> grid = program.get<std::vector<size_t>>("--grid");
        NGRIDS = {grid[0], grid[1], grid[2]};  // set global variable
    }

    auto input = program.get<std::string>("input");
    std::string output_stem = input;
    if(program.present("-o"))
    {
        output_stem = program.get<std::string>("-o");
    }

    int projection_id = get_projection_id(program);

    // only if found ".stl", ".off"  must be converted to stl file
    if(input.find(".stl") !=std::string::npos)
    {
        BoundBoxType bbox;
        if (true)  // program.present("--bbox")
        {
            bbox = program.get<BoundBoxType>("--bbox");
            //auto trsf = generate_transformation();  translation will also change boundbox
            project_mesh(input, output_stem, projection_id, bbox);
        }
    }
    else
    {
        project_geom(input, output_stem, projection_id, program.get<bool>("--bop"));
        // if(program.get<bool>("--bop"))
        // {
        //     //bop(input, output_stem);  // working but extremely slow, can be used to compare speed
        //     project(input, output_stem, false, program.get<bool>("--bop"));
        // }
        // else
        // {
        //     project(input, output_stem);   
        // }
    }
    
    return 0;
}