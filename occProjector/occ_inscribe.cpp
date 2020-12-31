#include "occ_inscribe.h"

#include <argparse.hpp>


int bop(std::string input, const std::string& output_file_stem)
{
    Bnd_OBB obb;
    auto shape = prepare_shape(input, obb);
    Bnd_Box box;
    BRepBndLib::Add(shape, box);

    // BOP OBB subtract thisShape

    // get the fragment with the max volume if not single, may also worth know the second largest

    // use other methods to calc inscribed shape

    return 0;

}


int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("occ_inscribe");

  program.add_argument("input")
    .required()
    .help("input geometry file path");

  program.add_argument("-o", "--output")
    .help("specify the output file stem");

  program.add_argument("--grid").nargs(3)
    .help("image pixel for x, y, z as an integer array")
    .default_value(std::vector<int>{64, 64, 64})
    .action([](const std::string& value) { return std::stoi(value); });

  program.add_argument("--bop")
    .help("use the very slow BOP method")
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
  
    //test_IndexedMap();
    auto input = program.get<std::string>("input");
    std::string output_stem = input;
    if(program.present("-o"))
    {
        output_stem = program.get<std::string>("-o");
    }

    if(program.get<bool>("--bop"))
    {
        bop(input, output_stem);  // working but extremely slow, can be used to compare speed
    }
    else
    {
        inscribe(input, output_stem);
    }
    
    return 0;
}