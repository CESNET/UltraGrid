#ifndef ASTAT_H_64f15ac8bc64

#ifdef __cplusplus
extern "C" {
#endif
struct ug_connection;

struct ug_connection *ug_control_connection_init(int local_port);
bool ug_control_get_volumes(struct ug_connection *c, double peak[], double rms[], int *count);
void ug_control_connection_done(struct ug_connection *);

void ug_control_init();
void ug_control_cleanup();

bool astat_parse_line(const char *str, double volpeak[2], double volrms[2]);

#ifdef __cplusplus
}
#endif

#endif
