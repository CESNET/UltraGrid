# R12L identity-mapped HEVC path

## Purpose

This path transports a DeckLink `R12L` signal through a hardware HEVC
4:4:4 10-bit encoder without applying an RGB-to-Y'CbCr matrix. It is intended
for low-latency, full-range RGB workflows in which preserving the three RGB
components is more important than producing a conventionally interpreted YUV
picture.

The key idea is to use a 10-bit 4:4:4 HEVC surface as a three-component
container:

| HEVC/Y410 component | Carried value |
| --- | --- |
| Y | G |
| U | B |
| V | R |

The values are identity-mapped, not color-matrix converted. A receiver that
knows this convention reverses the mapping and reconstructs RGB. A generic
YUV decoder or display will not produce correct colors because it will
interpret the components as ordinary luma and chroma.

## Signal path

### Encoder

1. DeckLink captures 12-bit packed RGB as `R12L`.
2. DeckLink 16.x buffers are accessed through `IDeckLinkVideoBuffer` and stay
   pinned until downstream compression releases the frame.
3. An OpenCL kernel unpacks each 12-bit RGB component and scales it to 10 bits
   using full-range integer rounding:

   ```
   value10 = round(value12 * 1023 / 4095)
   ```

4. When Intel OpenCL/VA-API surface sharing is available, the kernel writes
   `Y=G`, `U=B`, and `V=R` directly into the encoder's Y410-compatible VA
   surface.
5. VA-API encodes that hardware surface as HEVC Main 4:4:4 10. If surface
   sharing is unavailable, the encoder falls back to an `XV30` software
   surface followed by the normal VA-API hardware-frame upload.

Using an `XV30` software surface is deliberate. Supplying an RGB surface to
the VA-API encoder can cause the driver to apply an RGB-to-YUV conversion.
The identity-packed surface gives the hardware encoder the three component
planes directly.

### Receiver

1. libavcodec decodes HEVC, optionally with Intel QSV, to `XV30`.
2. An OpenCL kernel interprets the packed components as the identity mapping,
   extracts `R=V`, `G=Y`, and `B=U`, and writes either:
   - `R10k`, for a lower-bandwidth 10-bit RGB DeckLink output path; or
   - `R12L`, scaling each 10-bit component back to 12 bits.
3. DeckLink outputs the reconstructed RGB signal. The DeckLink display path
   can explicitly request 4:4:4 SDI output and single-link SDI operation.

The 10-to-12-bit reconstruction is:

```
value12 = round(value10 * 4095 / 1023)
```

The round trip is therefore component preserving at 10-bit precision. It
cannot recreate the two least-significant bits discarded by the 12-to-10-bit
quantization.

## Color range and metadata

For the identity path, the libavcodec encoder context is marked as:

- RGB matrix/identity (`AVCOL_SPC_RGB`)
- full range (`AVCOL_RANGE_JPEG`)
- BT.709 primaries
- BT.709 transfer characteristic

BT.709 here describes the source RGB primaries and transfer function. It does
not mean that the identity-mapped components have undergone a Rec.709
RGB-to-Y'CbCr matrix.

On R10k output, `--param bmd-r10k-full-range` prevents UltraGrid's normal
full-to-limited R10k conversion. Both endpoints must agree that the HEVC 4:4:4
components carry identity-mapped, full-range RGB.

## Latency controls

The path adds several controls used by the tested low-latency configuration:

- QSV decoder asynchronous depth is set to one.
- `--param low-latency-video` removes UltraGrid's default one-frame RTP video
  playout delay. This is suitable only for reliable, low-jitter networks.
- DeckLink synchronized output may be used to absorb small decoder timing
  variation and avoid unscheduled repeat/tear behavior.
- Repeated DeckLink frames retain their COM reference until the scheduling
  API releases them, preventing use-after-release artifacts.

Reducing these queues trades resilience for latency. A depth that works on
one GPU, format, bitrate, and network is not automatically safe for another.

## Audio timing

Audio remains independent of the identity video mapping. The patch preserves
the input RTP timestamp on decoded audio frames; the previous assignment
wrote the timestamp back to the input object instead of the decoded output.
This matters when DeckLink uses timestamped audio with synchronized video
playback.

If a source genuinely supplies no audio timestamp, the DeckLink display
fallback appends samples to the current scheduled audio timeline instead of
combining the `-1` sentinel with the video RTP epoch.

PCM and Opus can both use this timing path. Codec selection, Opus bitrate,
frame duration, and decoder selection remain runtime configuration rather
than properties of the R12L transport.

## Current implementation boundaries

- The identity mapping is private to cooperating endpoints; it is not a
  standard YUV representation.
- The hardware codec must support HEVC 4:4:4 10-bit surfaces.
- OpenCL is currently linked into the libavcodec compression and
  decompression modules when those modules are enabled.
- The encoder avoids the intermediate software surface and second upload when
  Intel OpenCL/VA-API surface sharing is available. DeckLink capture is still
  CPU-addressable memory and requires one upload to OpenCL, so this is a
  single-copy encoder path rather than end-to-end zero-copy.
- The receiver's QSV asynchronous depth is currently fixed at one for QSV
  decoders, favoring latency over general-purpose robustness.
- `R12L` output has 12-bit packing but only 10 bits of source precision after
  HEVC transport. `R10k` avoids that expansion and reduces output-side memory
  traffic.

## Example configuration

The exact device identifiers and transport destination are site-specific.
The essential video options are:

```sh
# Sender
uv -t decklink:codec=R12L \
   -c libavcodec:encoder=hevc_vaapi:rgb:depth=10:subsampling=444 \
   RECEIVER

# Receiver
uv -d decklink:single-link:synchronized=3 \
   --param decoder-use-codec=R10k \
   --param bmd-r10k-full-range \
   --param force-lavd-decoder=hevc_qsv \
   --param low-latency-video
```

Bitrate, GOP structure, audio codec, and queue depths should be selected for
the deployment rather than treated as part of the identity mapping itself.

## Validation

The implementation adds conversion coverage for `R12L` in the libavcodec
conversion tests. It has also been exercised end-to-end with UHD 2160p24
12-bit RGB DeckLink input, HEVC Main 4:4:4 10 hardware encoding, QSV hardware
decoding, and single-link 4:4:4 DeckLink output.
