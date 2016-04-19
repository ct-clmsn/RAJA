//AUTOGENERATED BY gen_forallN_generic.py
/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */
  
#ifndef RAJA_forallN_generic_HXX__
#define RAJA_forallN_generic_HXX__

#include "forallN_generic_lf.hxx"

namespace RAJA {


/*!
 * \brief Provides abstraction of a 1-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
template<typename POLICY, typename IdxI=Index_type, typename TI, typename BODY>
RAJA_INLINE
void forallN(TI const is_i, BODY const body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI> IDX_CONV;
  IDX_CONV lamb(body);

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i));
}

/*!
 * \brief Provides abstraction of a 2-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
template<typename POLICY, typename IdxI=Index_type, typename IdxJ=Index_type, typename TI, typename TJ, typename BODY>
RAJA_INLINE
void forallN(TI const is_i, TJ const is_j, BODY const body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  using PolicyJ = typename std::tuple_element<1, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI, IdxJ> IDX_CONV;
  IDX_CONV lamb(body);

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i),
    ForallN_PolicyPair<PolicyJ, TJ>(is_j));
}

/*!
 * \brief Provides abstraction of a 3-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
template<typename POLICY, typename IdxI=Index_type, typename IdxJ=Index_type, typename IdxK=Index_type, typename TI, typename TJ, typename TK, typename BODY>
RAJA_INLINE
void forallN(TI const is_i, TJ const is_j, TK const is_k, BODY const body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  using PolicyJ = typename std::tuple_element<1, typename ExecPolicies::tuple>::type;
  using PolicyK = typename std::tuple_element<2, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI, IdxJ, IdxK> IDX_CONV;
  IDX_CONV lamb(body);

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i),
    ForallN_PolicyPair<PolicyJ, TJ>(is_j),
    ForallN_PolicyPair<PolicyK, TK>(is_k));
}

/*!
 * \brief Provides abstraction of a 4-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
template<typename POLICY, typename IdxI=Index_type, typename IdxJ=Index_type, typename IdxK=Index_type, typename IdxL=Index_type, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE
void forallN(TI const is_i, TJ const is_j, TK const is_k, TL const is_l, BODY const body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  using PolicyJ = typename std::tuple_element<1, typename ExecPolicies::tuple>::type;
  using PolicyK = typename std::tuple_element<2, typename ExecPolicies::tuple>::type;
  using PolicyL = typename std::tuple_element<3, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI, IdxJ, IdxK, IdxL> IDX_CONV;
  IDX_CONV lamb(body);

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i),
    ForallN_PolicyPair<PolicyJ, TJ>(is_j),
    ForallN_PolicyPair<PolicyK, TK>(is_k),
    ForallN_PolicyPair<PolicyL, TL>(is_l));
}

/*!
 * \brief Provides abstraction of a 5-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
template<typename POLICY, typename IdxI=Index_type, typename IdxJ=Index_type, typename IdxK=Index_type, typename IdxL=Index_type, typename IdxM=Index_type, typename TI, typename TJ, typename TK, typename TL, typename TM, typename BODY>
RAJA_INLINE
void forallN(TI const is_i, TJ const is_j, TK const is_k, TL const is_l, TM const is_m, BODY const body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  using PolicyJ = typename std::tuple_element<1, typename ExecPolicies::tuple>::type;
  using PolicyK = typename std::tuple_element<2, typename ExecPolicies::tuple>::type;
  using PolicyL = typename std::tuple_element<3, typename ExecPolicies::tuple>::type;
  using PolicyM = typename std::tuple_element<4, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI, IdxJ, IdxK, IdxL, IdxM> IDX_CONV;
  IDX_CONV lamb(body);

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i),
    ForallN_PolicyPair<PolicyJ, TJ>(is_j),
    ForallN_PolicyPair<PolicyK, TK>(is_k),
    ForallN_PolicyPair<PolicyL, TL>(is_l),
    ForallN_PolicyPair<PolicyM, TM>(is_m));
}



} // namespace RAJA
  
#endif

