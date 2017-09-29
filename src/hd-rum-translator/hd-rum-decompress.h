#ifdef __cplusplus
extern "C" {
#endif

struct module;

enum hd_rum_mode_t {NORMAL, BLEND, CONFERENCE};
struct hd_rum_output_conf{
        enum hd_rum_mode_t mode;
        const char *arg;
};

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count);
void *hd_rum_decompress_init(struct module *parent, struct hd_rum_output_conf conf, const char *capture_filter);
void hd_rum_decompress_done(void *state);
void hd_rum_decompress_set_active(void *decompress_state, void *recompress_state, bool active);
void hd_rum_decompress_remove_port(void *decompress_state, int index);
void hd_rum_decompress_append_port(void *decompress_state, void *recompress_state);
int hd_rum_decompress_get_num_active_ports(void *decompress_state);

#ifdef __cplusplus
}
#endif

struct video_frame;

#ifdef __cplusplus
class frame_recv_delegate {
        public:
                virtual void frame_arrived(struct video_frame *) = 0;
};
#endif

