
ä
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8°ü
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
n
Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_1
g
!Adamax/beta_1/Read/ReadVariableOpReadVariableOpAdamax/beta_1*
_output_shapes
: *
dtype0
n
Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_2
g
!Adamax/beta_2/Read/ReadVariableOpReadVariableOpAdamax/beta_2*
_output_shapes
: *
dtype0
l
Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/decay
e
 Adamax/decay/Read/ReadVariableOpReadVariableOpAdamax/decay*
_output_shapes
: *
dtype0
|
Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdamax/learning_rate
u
(Adamax/learning_rate/Read/ReadVariableOpReadVariableOpAdamax/learning_rate*
_output_shapes
: *
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adamax/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdamax/dense_2/kernel/m

+Adamax/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_2/kernel/m*
_output_shapes
:	*
dtype0

Adamax/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/dense_2/bias/m
{
)Adamax/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_2/bias/m*
_output_shapes
:*
dtype0

Adamax/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdamax/dense_3/kernel/m

+Adamax/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_3/kernel/m*
_output_shapes
:	*
dtype0

Adamax/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/dense_3/bias/m
|
)Adamax/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_3/bias/m*
_output_shapes	
:*
dtype0

Adamax/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdamax/dense_2/kernel/v

+Adamax/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_2/kernel/v*
_output_shapes
:	*
dtype0

Adamax/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/dense_2/bias/v
{
)Adamax/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_2/bias/v*
_output_shapes
:*
dtype0

Adamax/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdamax/dense_3/kernel/v

+Adamax/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_3/kernel/v*
_output_shapes
:	*
dtype0

Adamax/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/dense_3/bias/v
|
)Adamax/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_3/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Õ#
valueË#BÈ# BÁ#
Ì
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 


layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratemWmXmYmZv[v\v]v^

0
1
2
3
 

0
1
2
3
­

layers
 layer_metrics
!layer_regularization_losses
trainable_variables
regularization_losses
"non_trainable_variables
#metrics
	variables
 
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

kernel
bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api

0
1
 

0
1
­

,layers
-layer_metrics
.layer_regularization_losses
trainable_variables
regularization_losses
/non_trainable_variables
0metrics
	variables
h

kernel
bias
1trainable_variables
2regularization_losses
3	variables
4	keras_api
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api

0
1
 

0
1
­

9layers
:layer_metrics
;layer_regularization_losses
trainable_variables
regularization_losses
<non_trainable_variables
=metrics
	variables
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEAdamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_2/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_2/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_3/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_3/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 

>0
 
 
 
­

?layers
@layer_metrics
Alayer_regularization_losses
$trainable_variables
%regularization_losses
Bnon_trainable_variables
Cmetrics
&	variables

0
1
 

0
1
­

Dlayers
Elayer_metrics
Flayer_regularization_losses
(trainable_variables
)regularization_losses
Gnon_trainable_variables
Hmetrics
*	variables


0
1
 
 
 
 

0
1
 

0
1
­

Ilayers
Jlayer_metrics
Klayer_regularization_losses
1trainable_variables
2regularization_losses
Lnon_trainable_variables
Mmetrics
3	variables
 
 
 
­

Nlayers
Olayer_metrics
Player_regularization_losses
5trainable_variables
6regularization_losses
Qnon_trainable_variables
Rmetrics
7	variables

0
1
 
 
 
 
4
	Stotal
	Tcount
U	variables
V	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

U	variables
yw
VARIABLE_VALUEAdamax/dense_2/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/dense_2/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/dense_3/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/dense_3/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/dense_2/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/dense_2/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/dense_3/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/dense_3/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_6Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
ú
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6dense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_139885
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdamax/iter/Read/ReadVariableOp!Adamax/beta_1/Read/ReadVariableOp!Adamax/beta_2/Read/ReadVariableOp Adamax/decay/Read/ReadVariableOp(Adamax/learning_rate/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adamax/dense_2/kernel/m/Read/ReadVariableOp)Adamax/dense_2/bias/m/Read/ReadVariableOp+Adamax/dense_3/kernel/m/Read/ReadVariableOp)Adamax/dense_3/bias/m/Read/ReadVariableOp+Adamax/dense_2/kernel/v/Read/ReadVariableOp)Adamax/dense_2/bias/v/Read/ReadVariableOp+Adamax/dense_3/kernel/v/Read/ReadVariableOp)Adamax/dense_3/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_140210

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameAdamax/iterAdamax/beta_1Adamax/beta_2Adamax/decayAdamax/learning_ratedense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotalcountAdamax/dense_2/kernel/mAdamax/dense_2/bias/mAdamax/dense_3/kernel/mAdamax/dense_3/bias/mAdamax/dense_2/kernel/vAdamax/dense_2/bias/vAdamax/dense_3/kernel/vAdamax/dense_3/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_140277ã¢
È
F
*__inference_reshape_1_layer_call_fn_140117

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1396712
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö	
ö
C__inference_dense_3_layer_call_and_return_conditional_losses_139652

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
(__inference_model_1_layer_call_fn_139911

inputs
unknown:	
	unknown_0:
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1398122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
õ
C__inference_dense_2_layer_call_and_return_conditional_losses_140093

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ï
(__inference_model_1_layer_call_fn_139783
input_6
unknown:	
	unknown_0:
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1397722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
­

-__inference_sequential_3_layer_call_fn_140016

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1396742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
±
C__inference_model_1_layer_call_and_return_conditional_losses_139965

inputsF
3sequential_2_dense_2_matmul_readvariableop_resource:	B
4sequential_2_dense_2_biasadd_readvariableop_resource:F
3sequential_3_dense_3_matmul_readvariableop_resource:	C
4sequential_3_dense_3_biasadd_readvariableop_resource:	
identity¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_3/dense_3/BiasAdd/ReadVariableOp¢*sequential_3/dense_3/MatMul/ReadVariableOp
sequential_2/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential_2/flatten_1/Const­
sequential_2/flatten_1/ReshapeReshapeinputs%sequential_2/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_2/flatten_1/ReshapeÍ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_2/dense_2/MatMul/ReadVariableOpÓ
sequential_2/dense_2/MatMulMatMul'sequential_2/flatten_1/Reshape:output:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_2/MatMulË
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOpÕ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_2/BiasAddÍ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÒ
sequential_3/dense_3/MatMulMatMul%sequential_2/dense_2/BiasAdd:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_3/BiasAdd
sequential_3/reshape_1/ShapeShape%sequential_3/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_3/reshape_1/Shape¢
*sequential_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_3/reshape_1/strided_slice/stack¦
,sequential_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_1/strided_slice/stack_1¦
,sequential_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_1/strided_slice/stack_2ì
$sequential_3/reshape_1/strided_sliceStridedSlice%sequential_3/reshape_1/Shape:output:03sequential_3/reshape_1/strided_slice/stack:output:05sequential_3/reshape_1/strided_slice/stack_1:output:05sequential_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_3/reshape_1/strided_slice
&sequential_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_1/Reshape/shape/1
&sequential_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_1/Reshape/shape/2
$sequential_3/reshape_1/Reshape/shapePack-sequential_3/reshape_1/strided_slice:output:0/sequential_3/reshape_1/Reshape/shape/1:output:0/sequential_3/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/reshape_1/Reshape/shape×
sequential_3/reshape_1/ReshapeReshape%sequential_3/dense_3/BiasAdd:output:0-sequential_3/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_3/reshape_1/Reshapeµ
IdentityIdentity'sequential_3/reshape_1/Reshape:output:0,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
F
*__inference_flatten_1_layer_call_fn_140068

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1395362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö	
ö
C__inference_dense_3_layer_call_and_return_conditional_losses_140112

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

-__inference_sequential_2_layer_call_fn_139562
input_4
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Æ

H__inference_sequential_3_layer_call_and_return_conditional_losses_140044

inputs9
&dense_3_matmul_readvariableop_resource:	6
'dense_3_biasadd_readvariableop_resource:	
identity¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddj
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2Ò
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape£
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape_1/Reshape³
IdentityIdentityreshape_1/Reshape:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

H__inference_sequential_2_layer_call_and_return_conditional_losses_140007

inputs9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_1/Const
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAdd­
IdentityIdentitydense_2/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ë
H__inference_sequential_3_layer_call_and_return_conditional_losses_139718

inputs!
dense_3_139711:	
dense_3_139713:	
identity¢dense_3/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_139711dense_3_139713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1396522!
dense_3/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1396712
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

-__inference_sequential_3_layer_call_fn_139681
input_5
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1396742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5

Ì
H__inference_sequential_3_layer_call_and_return_conditional_losses_139754
input_5!
dense_3_139747:	
dense_3_139749:	
identity¢dense_3/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_3_139747dense_3_139749*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1396522!
dense_3/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1396712
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5

È
C__inference_model_1_layer_call_and_return_conditional_losses_139850
input_6&
sequential_2_139839:	!
sequential_2_139841:&
sequential_3_139844:	"
sequential_3_139846:	
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall©
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_6sequential_2_139839sequential_2_139841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395552&
$sequential_2/StatefulPartitionedCallÓ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_139844sequential_3_139846*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1396742&
$sequential_3/StatefulPartitionedCallÓ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
±

H__inference_sequential_2_layer_call_and_return_conditional_losses_139995

inputs9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_1/Const
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_1/Reshape¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAdd­
IdentityIdentitydense_2/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
0

__inference__traced_save_140210
file_prefix*
&savev2_adamax_iter_read_readvariableop	,
(savev2_adamax_beta_1_read_readvariableop,
(savev2_adamax_beta_2_read_readvariableop+
'savev2_adamax_decay_read_readvariableop3
/savev2_adamax_learning_rate_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adamax_dense_2_kernel_m_read_readvariableop4
0savev2_adamax_dense_2_bias_m_read_readvariableop6
2savev2_adamax_dense_3_kernel_m_read_readvariableop4
0savev2_adamax_dense_3_bias_m_read_readvariableop6
2savev2_adamax_dense_2_kernel_v_read_readvariableop4
0savev2_adamax_dense_2_bias_v_read_readvariableop6
2savev2_adamax_dense_3_kernel_v_read_readvariableop4
0savev2_adamax_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameª

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¼	
value²	B¯	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices§
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_adamax_iter_read_readvariableop(savev2_adamax_beta_1_read_readvariableop(savev2_adamax_beta_2_read_readvariableop'savev2_adamax_decay_read_readvariableop/savev2_adamax_learning_rate_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adamax_dense_2_kernel_m_read_readvariableop0savev2_adamax_dense_2_bias_m_read_readvariableop2savev2_adamax_dense_3_kernel_m_read_readvariableop0savev2_adamax_dense_3_bias_m_read_readvariableop2savev2_adamax_dense_2_kernel_v_read_readvariableop0savev2_adamax_dense_2_bias_v_read_readvariableop2savev2_adamax_dense_3_kernel_v_read_readvariableop0savev2_adamax_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes}
{: : : : : : :	::	:: : :	::	::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!	

_output_shapes	
::


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::

_output_shapes
: 

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_139671

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
(__inference_model_1_layer_call_fn_139898

inputs
unknown:	
	unknown_0:
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1397722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ï
(__inference_model_1_layer_call_fn_139836
input_6
unknown:	
	unknown_0:
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1398122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
ß
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_139536

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_dense_3_layer_call_fn_140102

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1396522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
õ
C__inference_dense_2_layer_call_and_return_conditional_losses_139548

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ê
H__inference_sequential_2_layer_call_and_return_conditional_losses_139599

inputs!
dense_2_139593:	
dense_2_139595:
identity¢dense_2/StatefulPartitionedCallØ
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1395362
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_139593dense_2_139595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1395482!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_dense_2_layer_call_fn_140083

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1395482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ë
H__inference_sequential_3_layer_call_and_return_conditional_losses_139674

inputs!
dense_3_139653:	
dense_3_139655:	
identity¢dense_3/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_139653dense_3_139655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1396522!
dense_3/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1396712
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

-__inference_sequential_2_layer_call_fn_139615
input_4
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4

Ê
H__inference_sequential_2_layer_call_and_return_conditional_losses_139555

inputs!
dense_2_139549:	
dense_2_139551:
identity¢dense_2/StatefulPartitionedCallØ
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1395362
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_139549dense_2_139551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1395482!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½S
·
"__inference__traced_restore_140277
file_prefix&
assignvariableop_adamax_iter:	 *
 assignvariableop_1_adamax_beta_1: *
 assignvariableop_2_adamax_beta_2: )
assignvariableop_3_adamax_decay: 1
'assignvariableop_4_adamax_learning_rate: 4
!assignvariableop_5_dense_2_kernel:	-
assignvariableop_6_dense_2_bias:4
!assignvariableop_7_dense_3_kernel:	.
assignvariableop_8_dense_3_bias:	"
assignvariableop_9_total: #
assignvariableop_10_count: >
+assignvariableop_11_adamax_dense_2_kernel_m:	7
)assignvariableop_12_adamax_dense_2_bias_m:>
+assignvariableop_13_adamax_dense_3_kernel_m:	8
)assignvariableop_14_adamax_dense_3_bias_m:	>
+assignvariableop_15_adamax_dense_2_kernel_v:	7
)assignvariableop_16_adamax_dense_2_bias_v:>
+assignvariableop_17_adamax_dense_3_kernel_v:	8
)assignvariableop_18_adamax_dense_3_bias_v:	
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9°

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¼	
value²	B¯	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adamax_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_adamax_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_adamax_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_adamax_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_adamax_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¤
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11³
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adamax_dense_2_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adamax_dense_2_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13³
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adamax_dense_3_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adamax_dense_3_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15³
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adamax_dense_2_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adamax_dense_2_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17³
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adamax_dense_3_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18±
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adamax_dense_3_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19ó
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ý
Ë
$__inference_signature_wrapper_139885
input_6
unknown:	
	unknown_0:
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1395232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6

Ë
H__inference_sequential_2_layer_call_and_return_conditional_losses_139635
input_4!
dense_2_139629:	
dense_2_139631:
identity¢dense_2/StatefulPartitionedCallÙ
flatten_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1395362
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_139629dense_2_139631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1395482!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Æ

H__inference_sequential_3_layer_call_and_return_conditional_losses_140063

inputs9
&dense_3_matmul_readvariableop_resource:	6
'dense_3_biasadd_readvariableop_resource:	
identity¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddj
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2Ò
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape£
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape_1/Reshape³
IdentityIdentityreshape_1/Reshape:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ë
H__inference_sequential_2_layer_call_and_return_conditional_losses_139625
input_4!
dense_2_139619:	
dense_2_139621:
identity¢dense_2/StatefulPartitionedCallÙ
flatten_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1395362
flatten_1/PartitionedCall«
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_139619dense_2_139621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1395482!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
­

-__inference_sequential_3_layer_call_fn_140025

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1397182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ì
H__inference_sequential_3_layer_call_and_return_conditional_losses_139744
input_5!
dense_3_139737:	
dense_3_139739:	
identity¢dense_3/StatefulPartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_3_139737dense_3_139739*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1396522!
dense_3/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1396712
reshape_1/PartitionedCall
IdentityIdentity"reshape_1/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_140130

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ç
C__inference_model_1_layer_call_and_return_conditional_losses_139772

inputs&
sequential_2_139761:	!
sequential_2_139763:&
sequential_3_139766:	"
sequential_3_139768:	
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¨
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_139761sequential_2_139763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395552&
$sequential_2/StatefulPartitionedCallÓ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_139766sequential_3_139768*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1396742&
$sequential_3/StatefulPartitionedCallÓ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

-__inference_sequential_2_layer_call_fn_139974

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_140074

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ç
C__inference_model_1_layer_call_and_return_conditional_losses_139812

inputs&
sequential_2_139801:	!
sequential_2_139803:&
sequential_3_139806:	"
sequential_3_139808:	
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¨
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_139801sequential_2_139803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395992&
$sequential_2/StatefulPartitionedCallÓ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_139806sequential_3_139808*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1397182&
$sequential_3/StatefulPartitionedCallÓ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
.
Ð
!__inference__wrapped_model_139523
input_6N
;model_1_sequential_2_dense_2_matmul_readvariableop_resource:	J
<model_1_sequential_2_dense_2_biasadd_readvariableop_resource:N
;model_1_sequential_3_dense_3_matmul_readvariableop_resource:	K
<model_1_sequential_3_dense_3_biasadd_readvariableop_resource:	
identity¢3model_1/sequential_2/dense_2/BiasAdd/ReadVariableOp¢2model_1/sequential_2/dense_2/MatMul/ReadVariableOp¢3model_1/sequential_3/dense_3/BiasAdd/ReadVariableOp¢2model_1/sequential_3/dense_3/MatMul/ReadVariableOp
$model_1/sequential_2/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2&
$model_1/sequential_2/flatten_1/ConstÆ
&model_1/sequential_2/flatten_1/ReshapeReshapeinput_6-model_1/sequential_2/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_1/sequential_2/flatten_1/Reshapeå
2model_1/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2model_1/sequential_2/dense_2/MatMul/ReadVariableOpó
#model_1/sequential_2/dense_2/MatMulMatMul/model_1/sequential_2/flatten_1/Reshape:output:0:model_1/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_1/sequential_2/dense_2/MatMulã
3model_1/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_1/sequential_2/dense_2/BiasAdd/ReadVariableOpõ
$model_1/sequential_2/dense_2/BiasAddBiasAdd-model_1/sequential_2/dense_2/MatMul:product:0;model_1/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_1/sequential_2/dense_2/BiasAddå
2model_1/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp;model_1_sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2model_1/sequential_3/dense_3/MatMul/ReadVariableOpò
#model_1/sequential_3/dense_3/MatMulMatMul-model_1/sequential_2/dense_2/BiasAdd:output:0:model_1/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_1/sequential_3/dense_3/MatMulä
3model_1/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_1_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_1/sequential_3/dense_3/BiasAdd/ReadVariableOpö
$model_1/sequential_3/dense_3/BiasAddBiasAdd-model_1/sequential_3/dense_3/MatMul:product:0;model_1/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_1/sequential_3/dense_3/BiasAdd©
$model_1/sequential_3/reshape_1/ShapeShape-model_1/sequential_3/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2&
$model_1/sequential_3/reshape_1/Shape²
2model_1/sequential_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_1/sequential_3/reshape_1/strided_slice/stack¶
4model_1/sequential_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_1/sequential_3/reshape_1/strided_slice/stack_1¶
4model_1/sequential_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_1/sequential_3/reshape_1/strided_slice/stack_2
,model_1/sequential_3/reshape_1/strided_sliceStridedSlice-model_1/sequential_3/reshape_1/Shape:output:0;model_1/sequential_3/reshape_1/strided_slice/stack:output:0=model_1/sequential_3/reshape_1/strided_slice/stack_1:output:0=model_1/sequential_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_1/sequential_3/reshape_1/strided_slice¢
.model_1/sequential_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.model_1/sequential_3/reshape_1/Reshape/shape/1¢
.model_1/sequential_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.model_1/sequential_3/reshape_1/Reshape/shape/2»
,model_1/sequential_3/reshape_1/Reshape/shapePack5model_1/sequential_3/reshape_1/strided_slice:output:07model_1/sequential_3/reshape_1/Reshape/shape/1:output:07model_1/sequential_3/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2.
,model_1/sequential_3/reshape_1/Reshape/shape÷
&model_1/sequential_3/reshape_1/ReshapeReshape-model_1/sequential_3/dense_3/BiasAdd:output:05model_1/sequential_3/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_1/sequential_3/reshape_1/ReshapeÝ
IdentityIdentity/model_1/sequential_3/reshape_1/Reshape:output:04^model_1/sequential_2/dense_2/BiasAdd/ReadVariableOp3^model_1/sequential_2/dense_2/MatMul/ReadVariableOp4^model_1/sequential_3/dense_3/BiasAdd/ReadVariableOp3^model_1/sequential_3/dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2j
3model_1/sequential_2/dense_2/BiasAdd/ReadVariableOp3model_1/sequential_2/dense_2/BiasAdd/ReadVariableOp2h
2model_1/sequential_2/dense_2/MatMul/ReadVariableOp2model_1/sequential_2/dense_2/MatMul/ReadVariableOp2j
3model_1/sequential_3/dense_3/BiasAdd/ReadVariableOp3model_1/sequential_3/dense_3/BiasAdd/ReadVariableOp2h
2model_1/sequential_3/dense_3/MatMul/ReadVariableOp2model_1/sequential_3/dense_3/MatMul/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
)
±
C__inference_model_1_layer_call_and_return_conditional_losses_139938

inputsF
3sequential_2_dense_2_matmul_readvariableop_resource:	B
4sequential_2_dense_2_biasadd_readvariableop_resource:F
3sequential_3_dense_3_matmul_readvariableop_resource:	C
4sequential_3_dense_3_biasadd_readvariableop_resource:	
identity¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_3/dense_3/BiasAdd/ReadVariableOp¢*sequential_3/dense_3/MatMul/ReadVariableOp
sequential_2/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential_2/flatten_1/Const­
sequential_2/flatten_1/ReshapeReshapeinputs%sequential_2/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_2/flatten_1/ReshapeÍ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_2/dense_2/MatMul/ReadVariableOpÓ
sequential_2/dense_2/MatMulMatMul'sequential_2/flatten_1/Reshape:output:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_2/MatMulË
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOpÕ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_2/BiasAddÍ
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOpÒ
sequential_3/dense_3/MatMulMatMul%sequential_2/dense_2/BiasAdd:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_3/MatMulÌ
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOpÖ
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_3/dense_3/BiasAdd
sequential_3/reshape_1/ShapeShape%sequential_3/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_3/reshape_1/Shape¢
*sequential_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_3/reshape_1/strided_slice/stack¦
,sequential_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_1/strided_slice/stack_1¦
,sequential_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_3/reshape_1/strided_slice/stack_2ì
$sequential_3/reshape_1/strided_sliceStridedSlice%sequential_3/reshape_1/Shape:output:03sequential_3/reshape_1/strided_slice/stack:output:05sequential_3/reshape_1/strided_slice/stack_1:output:05sequential_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_3/reshape_1/strided_slice
&sequential_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_1/Reshape/shape/1
&sequential_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_3/reshape_1/Reshape/shape/2
$sequential_3/reshape_1/Reshape/shapePack-sequential_3/reshape_1/strided_slice:output:0/sequential_3/reshape_1/Reshape/shape/1:output:0/sequential_3/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/reshape_1/Reshape/shape×
sequential_3/reshape_1/ReshapeReshape%sequential_3/dense_3/BiasAdd:output:0-sequential_3/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_3/reshape_1/Reshapeµ
IdentityIdentity'sequential_3/reshape_1/Reshape:output:0,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

È
C__inference_model_1_layer_call_and_return_conditional_losses_139864
input_6&
sequential_2_139853:	!
sequential_2_139855:&
sequential_3_139858:	"
sequential_3_139860:	
identity¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall©
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_6sequential_2_139853sequential_2_139855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395992&
$sequential_2/StatefulPartitionedCallÓ
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_139858sequential_3_139860*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1397182&
$sequential_3/StatefulPartitionedCallÓ
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
°

-__inference_sequential_3_layer_call_fn_139734
input_5
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1397182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5
¬

-__inference_sequential_2_layer_call_fn_139983

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_1395992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
?
input_64
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿD
sequential_34
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Þð
¬-
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature"+
_tf_keras_networkê*{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["sequential_3", 1, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28]}, "float32", "input_6"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]], "shared_object_id": 12}], "input_layers": [["input_6", 0, 0]], "output_layers": [["sequential_3", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}


layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"Õ
_tf_keras_sequential¶{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28]}, "float32", "input_4"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "shared_object_id": 1}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}]}}}
×
layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_sequentialõ{"name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}}]}, "inbound_nodes": [[["sequential_2", 1, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5]}, "float32", "input_5"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}, "shared_object_id": 11}]}}}

iter

beta_1

beta_2
	decay
learning_ratemWmXmYmZv[v\v]v^"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ê

layers
 layer_metrics
!layer_regularization_losses
trainable_variables
regularization_losses
"non_trainable_variables
#metrics
	variables
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
fserving_default"
signature_map

$trainable_variables
%regularization_losses
&	variables
'	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layerì{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 15}}
Ð

kernel
bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
i__call__
*j&call_and_return_all_conditional_losses"«
_tf_keras_layer{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

,layers
-layer_metrics
.layer_regularization_losses
trainable_variables
regularization_losses
/non_trainable_variables
0metrics
	variables
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
Ï

kernel
bias
1trainable_variables
2regularization_losses
3	variables
4	keras_api
k__call__
*l&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}

5trainable_variables
6regularization_losses
7	variables
8	keras_api
m__call__
*n&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [28, 28]}}, "shared_object_id": 11}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

9layers
:layer_metrics
;layer_regularization_losses
trainable_variables
regularization_losses
<non_trainable_variables
=metrics
	variables
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adamax/iter
: (2Adamax/beta_1
: (2Adamax/beta_2
: (2Adamax/decay
: (2Adamax/learning_rate
!:	2dense_2/kernel
:2dense_2/bias
!:	2dense_3/kernel
:2dense_3/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

?layers
@layer_metrics
Alayer_regularization_losses
$trainable_variables
%regularization_losses
Bnon_trainable_variables
Cmetrics
&	variables
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Dlayers
Elayer_metrics
Flayer_regularization_losses
(trainable_variables
)regularization_losses
Gnon_trainable_variables
Hmetrics
*	variables
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Ilayers
Jlayer_metrics
Klayer_regularization_losses
1trainable_variables
2regularization_losses
Lnon_trainable_variables
Mmetrics
3	variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Nlayers
Olayer_metrics
Player_regularization_losses
5trainable_variables
6regularization_losses
Qnon_trainable_variables
Rmetrics
7	variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ô
	Stotal
	Tcount
U	variables
V	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 18}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
S0
T1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
(:&	2Adamax/dense_2/kernel/m
!:2Adamax/dense_2/bias/m
(:&	2Adamax/dense_3/kernel/m
": 2Adamax/dense_3/bias/m
(:&	2Adamax/dense_2/kernel/v
!:2Adamax/dense_2/bias/v
(:&	2Adamax/dense_3/kernel/v
": 2Adamax/dense_3/bias/v
î2ë
(__inference_model_1_layer_call_fn_139783
(__inference_model_1_layer_call_fn_139898
(__inference_model_1_layer_call_fn_139911
(__inference_model_1_layer_call_fn_139836À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_1_layer_call_and_return_conditional_losses_139938
C__inference_model_1_layer_call_and_return_conditional_losses_139965
C__inference_model_1_layer_call_and_return_conditional_losses_139850
C__inference_model_1_layer_call_and_return_conditional_losses_139864À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
!__inference__wrapped_model_139523º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
input_6ÿÿÿÿÿÿÿÿÿ
2ÿ
-__inference_sequential_2_layer_call_fn_139562
-__inference_sequential_2_layer_call_fn_139974
-__inference_sequential_2_layer_call_fn_139983
-__inference_sequential_2_layer_call_fn_139615À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_2_layer_call_and_return_conditional_losses_139995
H__inference_sequential_2_layer_call_and_return_conditional_losses_140007
H__inference_sequential_2_layer_call_and_return_conditional_losses_139625
H__inference_sequential_2_layer_call_and_return_conditional_losses_139635À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_sequential_3_layer_call_fn_139681
-__inference_sequential_3_layer_call_fn_140016
-__inference_sequential_3_layer_call_fn_140025
-__inference_sequential_3_layer_call_fn_139734À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_3_layer_call_and_return_conditional_losses_140044
H__inference_sequential_3_layer_call_and_return_conditional_losses_140063
H__inference_sequential_3_layer_call_and_return_conditional_losses_139744
H__inference_sequential_3_layer_call_and_return_conditional_losses_139754À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_139885input_6"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_flatten_1_layer_call_fn_140068¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_140074¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_140083¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_140093¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_140102¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_140112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_1_layer_call_fn_140117¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_1_layer_call_and_return_conditional_losses_140130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¢
!__inference__wrapped_model_139523}4¢1
*¢'
%"
input_6ÿÿÿÿÿÿÿÿÿ
ª "?ª<
:
sequential_3*'
sequential_3ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_2_layer_call_and_return_conditional_losses_140093]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_2_layer_call_fn_140083P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_3_layer_call_and_return_conditional_losses_140112]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_3_layer_call_fn_140102P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_flatten_1_layer_call_and_return_conditional_losses_140074]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_flatten_1_layer_call_fn_140068P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¶
C__inference_model_1_layer_call_and_return_conditional_losses_139850o<¢9
2¢/
%"
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¶
C__inference_model_1_layer_call_and_return_conditional_losses_139864o<¢9
2¢/
%"
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 µ
C__inference_model_1_layer_call_and_return_conditional_losses_139938n;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 µ
C__inference_model_1_layer_call_and_return_conditional_losses_139965n;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_model_1_layer_call_fn_139783b<¢9
2¢/
%"
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_1_layer_call_fn_139836b<¢9
2¢/
%"
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_1_layer_call_fn_139898a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_1_layer_call_fn_139911a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_reshape_1_layer_call_and_return_conditional_losses_140130]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_reshape_1_layer_call_fn_140117P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
H__inference_sequential_2_layer_call_and_return_conditional_losses_139625i<¢9
2¢/
%"
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
H__inference_sequential_2_layer_call_and_return_conditional_losses_139635i<¢9
2¢/
%"
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
H__inference_sequential_2_layer_call_and_return_conditional_losses_139995h;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
H__inference_sequential_2_layer_call_and_return_conditional_losses_140007h;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_2_layer_call_fn_139562\<¢9
2¢/
%"
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_2_layer_call_fn_139615\<¢9
2¢/
%"
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_2_layer_call_fn_139974[;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_2_layer_call_fn_139983[;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿµ
H__inference_sequential_3_layer_call_and_return_conditional_losses_139744i8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 µ
H__inference_sequential_3_layer_call_and_return_conditional_losses_139754i8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ´
H__inference_sequential_3_layer_call_and_return_conditional_losses_140044h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ´
H__inference_sequential_3_layer_call_and_return_conditional_losses_140063h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_3_layer_call_fn_139681\8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_3_layer_call_fn_139734\8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_3_layer_call_fn_140016[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_3_layer_call_fn_140025[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ±
$__inference_signature_wrapper_139885?¢<
¢ 
5ª2
0
input_6%"
input_6ÿÿÿÿÿÿÿÿÿ"?ª<
:
sequential_3*'
sequential_3ÿÿÿÿÿÿÿÿÿ