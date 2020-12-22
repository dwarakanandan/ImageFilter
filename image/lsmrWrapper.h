#pragma once

#ifdef USE_LSQR
#include "lsqrBase.h"

template <typename L1, typename L2>
class lsmrWrapper : public lsqrBase
{
public:

	lsmrWrapper(L1 &lambda1, L2 &lambda2) : lambda1(lambda1), lambda2(lambda2) {}
	virtual ~lsmrWrapper() {}

	void Aprod1(unsigned int m, unsigned int n, const double * x, double * y) const { lambda1(m, n, x, y); }

	void Aprod2(unsigned int m, unsigned int n, double * x, const double * y) const { lambda2(m, n, x, y); }

private:

	L1 lambda1;
	L2 lambda2;
};
#else
#include "lsmrBase.h"

template <typename L1, typename L2>
class lsmrWrapper : public lsmrBase
{
public:

	lsmrWrapper(L1 &lambda1, L2 &lambda2) : lambda1(lambda1), lambda2(lambda2) {}
	virtual ~lsmrWrapper() {}

	void Aprod1(unsigned int m, unsigned int n, const double * x, double * y) const { lambda1(m, n, x, y); }

	void Aprod2(unsigned int m, unsigned int n, double * x, const double * y) const { lambda2(m, n, x, y); }

private:

	L1 lambda1;
	L2 lambda2;
};
#endif
