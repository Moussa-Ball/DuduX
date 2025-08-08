#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

static inline long long file_size(const std::string& p) {
    struct stat st; if (stat(p.c_str(), &st) != 0) return -1; return st.st_size;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <path> <in> <hidden> <out>\n";
        return 2;
    }
    std::string path = argv[1];
    uint64_t in = std::stoull(argv[2]);
    uint64_t hidden = std::stoull(argv[3]);
    uint64_t out = std::stoull(argv[4]);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Cannot open file: " << path << "\n";
        return 3;
    }
    char magic[8] = {0};
    ifs.read(magic, 8);
    const char ref[8] = {'D','X','1','B','M','L','P','\0'};
    if (std::memcmp(magic, ref, 8) != 0) {
        std::cerr << "Bad magic" << "\n"; return 4;
    }
    uint64_t in_f=0, hid_f=0, out_f=0, pack=0;
    ifs.read(reinterpret_cast<char*>(&in_f), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&hid_f), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&out_f), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&pack), sizeof(uint64_t));
    if (in_f != in || hid_f != hidden || out_f != out) {
        std::cerr << "Dim mismatch: got ("<<in_f<<","<<hid_f<<","<<out_f<<")";
        std::cerr << " expected ("<<in<<","<<hidden<<","<<out<<")\n";
        return 5;
    }
    if (pack != 64) { std::cerr << "pack_bits != 64\n"; return 6; }
    // compute expected total size
    size_t w1_words = (in + 63) / 64;
    size_t w2_words = (hidden + 63) / 64;
    long long expect = 8 + 4*8 + (long long)hidden * (long long)w1_words * 8ll
                                 + (long long)out    * (long long)w2_words * 8ll;
    long long fsz = file_size(path);
    if (fsz < 0) { std::cerr << "stat failed\n"; return 7; }
    if (fsz != expect) {
        std::cerr << "Size mismatch: got "<< fsz << ", expect "<< expect <<"\n";
        return 8;
    }
    std::cout << "OK in="<<in<<" hidden="<<hidden<<" out="<<out<<" size="<<fsz<<"\n";
    return 0;
}
