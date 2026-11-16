// Prints LLPCombine's own combined cut string per bin (JSON lines), plus resolved luminosity.
#include "ConfigParser.h"
#include <iostream>
static std::string esc(const std::string& s){std::string o;for(char c:s){if(c=='"'||c=='\\')o+='\\';o+=c;}return o;}
int main(int argc,char**argv){
  ConfigParser p; if(!p.LoadConfig(argv[1])) return 1;
  const auto& cfg=p.GetConfig();
  std::cout<<"{\"__luminosity__\":"<<cfg.luminosity<<"}\n";
  for(const auto& b: cfg.bins) std::cout<<"{\"bin\":\""<<esc(b.name)<<"\",\"cuts\":\""<<esc(p.GetCombinedCuts(b.name))<<"\"}\n";
}
