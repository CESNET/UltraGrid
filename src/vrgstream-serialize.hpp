#ifndef VFGSTREAM_SERIALIZE_HPP_5DF43E1E_EB46_4927_AD07_39F8B7C86D97
#define VFGSTREAM_SERIALIZE_HPP_5DF43E1E_EB46_4927_AD07_39F8B7C86D97

#include <iostream>

std::ostream &operator<<(std::ostream &output, const ProjectionFovTan &obj);
std::ostream &operator<<(std::ostream &output, const Vector3 &obj);
std::ostream &operator<<(std::ostream &output, const Quaternion &obj);
std::ostream &operator<<(std::ostream &output, const Pose &obj);
std::ostream &operator<<(std::ostream &output, const RenderPacket &obj);

std::ostream &operator<<(std::ostream &output, const ProjectionFovTan &obj) {
        output << obj.left << "," << obj.right << "," << obj.top << "," << obj.bottom;
        return output;
}

std::ostream &operator<<(std::ostream &output, const Vector3 &obj) {
        output << obj.x << "," << obj.y << "," << obj.z;
        return output;
}

std::ostream &operator<<(std::ostream &output, const Quaternion &obj) {
        output << obj.x << "," << obj.y << "," << obj.z << "," << obj.z;
        return output;
}

std::ostream &operator<<(std::ostream &output, const Pose &obj) {
        output << obj.position << "," << obj.orientation;
        return output;
}

std::ostream &operator<<(std::ostream &output, const RenderPacket &obj) {
        output << obj.left_projection_fov << "," <<
                obj.left_view_pose << "," <<
                obj.right_projection_fov << "," <<
                obj.right_view_pose << "," <<
                obj.pix_width_eye << "," <<
                obj.pix_height_eye << "," <<
                obj.timepoint << "," <<
                obj.frame << "," <<
                obj.dx_row_pitch << "," <<
                obj.dx_row_pitch_uv;
        return output;
}

#endif // defined VFGSTREAM_SERIALIZE_HPP_5DF43E1E_EB46_4927_AD07_39F8B7C86D97

