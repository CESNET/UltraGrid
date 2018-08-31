#ifdef __cplusplus
extern "C" {
#endif
struct ug_connection;

struct ug_connection *ug_control_connection_init(int local_port);
bool ug_control_get_volumes(struct ug_connection *c, double peak[], double rms[], int *count);
void ug_control_connection_done(struct ug_connection *);

void ug_control_init();
void ug_control_cleanup();

#ifdef __cplusplus
}
#endif
