#ifndef UG_HOLE_PUNCH_STATUS_CODE_HPP
#define UG_HOLE_PUNCH_STATUS_CODE_HPP

enum class CompletionStatus{
	Success,
	UnexpectedDisconnect,
	GracefulDisconnect,
	MsgError,
	Error
};

#endif
