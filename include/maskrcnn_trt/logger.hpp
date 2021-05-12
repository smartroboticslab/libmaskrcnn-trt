// SPDX-FileCopyrightText: 2019 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LOGGER_H
#define LOGGER_H

#include "logging.hpp"

extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);

#endif // LOGGER_H

