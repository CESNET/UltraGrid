#ifdef __cplusplus
extern "C" {
#endif

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count);
void *hd_rum_decompress_init(unsigned short rx_port);
void hd_rum_decompress_done(void *state);
void hd_rum_decompress_add_port(void *decompress_state, void *recompress_state);
void hd_rum_decompress_add_inactive_port(void *decompress_state, void *recompress_state);

#ifdef __cplusplus
}
#endif
