#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ink2 {

class embedded_server {
public:
    struct impl;
    embedded_server();
    ~embedded_server();
    embedded_server(const embedded_server &) = delete;
    embedded_server & operator=(const embedded_server &) = delete;

    bool start(const std::vector<std::string> & args);
    bool write_line(const std::string & line);
    bool read_line(std::string & line);
    bool alive() const;
    void stop();

private:
    std::shared_ptr<impl> p;
};

bool embedded_active();
bool embedded_read_line(std::string & line);
void embedded_write_line(const std::string & line);
int engine_main(int argc, char ** argv);

}
