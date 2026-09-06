/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#include "aodvatk-dpd.h"

namespace ns3
{
namespace aodvatk
{

bool
DuplicatePacketDetection::IsDuplicate(Ptr<const Packet> p, const Ipv4Header& header)
{
    return m_idCache.IsDuplicate(header.GetSource(), p->GetUid());
}

void
DuplicatePacketDetection::SetLifetime(Time lifetime)
{
    m_idCache.SetLifetime(lifetime);
}

Time
DuplicatePacketDetection::GetLifetime() const
{
    return m_idCache.GetLifeTime();
}

} // namespace aodvatk
} // namespace ns3
