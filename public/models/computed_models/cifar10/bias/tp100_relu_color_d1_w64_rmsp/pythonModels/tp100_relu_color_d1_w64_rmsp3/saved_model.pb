��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108��
�
dense_Dense13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_namedense_Dense13/kernel
~
(dense_Dense13/kernel/Read/ReadVariableOpReadVariableOpdense_Dense13/kernel*
_output_shapes
:	�@*
dtype0
|
dense_Dense13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedense_Dense13/bias
u
&dense_Dense13/bias/Read/ReadVariableOpReadVariableOpdense_Dense13/bias*
_output_shapes
:@*
dtype0
�
dense_Dense14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*%
shared_namedense_Dense14/kernel
~
(dense_Dense14/kernel/Read/ReadVariableOpReadVariableOpdense_Dense14/kernel*
_output_shapes
:	@�*
dtype0
}
dense_Dense14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_namedense_Dense14/bias
v
&dense_Dense14/bias/Read/ReadVariableOpReadVariableOpdense_Dense14/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

	0

1
2
3
 

	0

1
2
3
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
layer_regularization_losses
trainable_variables
 
`^
VARIABLE_VALUEdense_Dense13/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_Dense13/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
layer_regularization_losses
trainable_variables
`^
VARIABLE_VALUEdense_Dense14/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_Dense14/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
 layer_regularization_losses
trainable_variables
 
 

0
1
 
 
 
 
 
 
 
 
 
�
#serving_default_dense_Dense13_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_dense_Dense13_inputdense_Dense13/kerneldense_Dense13/biasdense_Dense14/kerneldense_Dense14/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference_signature_wrapper_180
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(dense_Dense13/kernel/Read/ReadVariableOp&dense_Dense13/bias/Read/ReadVariableOp(dense_Dense14/kernel/Read/ReadVariableOp&dense_Dense14/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*%
f R
__inference__traced_save_306
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_Dense13/kerneldense_Dense13/biasdense_Dense14/kerneldense_Dense14/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_restore_330��
�	
�
F__inference_dense_Dense13_layer_call_and_return_conditional_losses_245

inputs.
*matmul_readvariableop_dense_dense13_kernel-
)biasadd_readvariableop_dense_dense13_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp*matmul_readvariableop_dense_dense13_kernel*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_dense_dense13_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_216

inputs<
8dense_dense13_matmul_readvariableop_dense_dense13_kernel;
7dense_dense13_biasadd_readvariableop_dense_dense13_bias<
8dense_dense14_matmul_readvariableop_dense_dense14_kernel;
7dense_dense14_biasadd_readvariableop_dense_dense14_bias
identity��$dense_Dense13/BiasAdd/ReadVariableOp�#dense_Dense13/MatMul/ReadVariableOp�$dense_Dense14/BiasAdd/ReadVariableOp�#dense_Dense14/MatMul/ReadVariableOp�
#dense_Dense13/MatMul/ReadVariableOpReadVariableOp8dense_dense13_matmul_readvariableop_dense_dense13_kernel*
_output_shapes
:	�@*
dtype02%
#dense_Dense13/MatMul/ReadVariableOp�
dense_Dense13/MatMulMatMulinputs+dense_Dense13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/MatMul�
$dense_Dense13/BiasAdd/ReadVariableOpReadVariableOp7dense_dense13_biasadd_readvariableop_dense_dense13_bias*
_output_shapes
:@*
dtype02&
$dense_Dense13/BiasAdd/ReadVariableOp�
dense_Dense13/BiasAddBiasAdddense_Dense13/MatMul:product:0,dense_Dense13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/BiasAdd�
dense_Dense13/ReluReludense_Dense13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/Relu�
#dense_Dense14/MatMul/ReadVariableOpReadVariableOp8dense_dense14_matmul_readvariableop_dense_dense14_kernel*
_output_shapes
:	@�*
dtype02%
#dense_Dense14/MatMul/ReadVariableOp�
dense_Dense14/MatMulMatMul dense_Dense13/Relu:activations:0+dense_Dense14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_Dense14/MatMul�
$dense_Dense14/BiasAdd/ReadVariableOpReadVariableOp7dense_dense14_biasadd_readvariableop_dense_dense14_bias*
_output_shapes	
:�*
dtype02&
$dense_Dense14/BiasAdd/ReadVariableOp�
dense_Dense14/BiasAddBiasAdddense_Dense14/MatMul:product:0,dense_Dense14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_Dense14/BiasAdd�
dense_Dense14/ReluReludense_Dense14/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_Dense14/Relu�
IdentityIdentity dense_Dense14/Relu:activations:0%^dense_Dense13/BiasAdd/ReadVariableOp$^dense_Dense13/MatMul/ReadVariableOp%^dense_Dense14/BiasAdd/ReadVariableOp$^dense_Dense14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2L
$dense_Dense13/BiasAdd/ReadVariableOp$dense_Dense13/BiasAdd/ReadVariableOp2J
#dense_Dense13/MatMul/ReadVariableOp#dense_Dense13/MatMul/ReadVariableOp2L
$dense_Dense14/BiasAdd/ReadVariableOp$dense_Dense14/BiasAdd/ReadVariableOp2J
#dense_Dense14/MatMul/ReadVariableOp#dense_Dense14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_198

inputs<
8dense_dense13_matmul_readvariableop_dense_dense13_kernel;
7dense_dense13_biasadd_readvariableop_dense_dense13_bias<
8dense_dense14_matmul_readvariableop_dense_dense14_kernel;
7dense_dense14_biasadd_readvariableop_dense_dense14_bias
identity��$dense_Dense13/BiasAdd/ReadVariableOp�#dense_Dense13/MatMul/ReadVariableOp�$dense_Dense14/BiasAdd/ReadVariableOp�#dense_Dense14/MatMul/ReadVariableOp�
#dense_Dense13/MatMul/ReadVariableOpReadVariableOp8dense_dense13_matmul_readvariableop_dense_dense13_kernel*
_output_shapes
:	�@*
dtype02%
#dense_Dense13/MatMul/ReadVariableOp�
dense_Dense13/MatMulMatMulinputs+dense_Dense13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/MatMul�
$dense_Dense13/BiasAdd/ReadVariableOpReadVariableOp7dense_dense13_biasadd_readvariableop_dense_dense13_bias*
_output_shapes
:@*
dtype02&
$dense_Dense13/BiasAdd/ReadVariableOp�
dense_Dense13/BiasAddBiasAdddense_Dense13/MatMul:product:0,dense_Dense13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/BiasAdd�
dense_Dense13/ReluReludense_Dense13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_Dense13/Relu�
#dense_Dense14/MatMul/ReadVariableOpReadVariableOp8dense_dense14_matmul_readvariableop_dense_dense14_kernel*
_output_shapes
:	@�*
dtype02%
#dense_Dense14/MatMul/ReadVariableOp�
dense_Dense14/MatMulMatMul dense_Dense13/Relu:activations:0+dense_Dense14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_Dense14/MatMul�
$dense_Dense14/BiasAdd/ReadVariableOpReadVariableOp7dense_dense14_biasadd_readvariableop_dense_dense14_bias*
_output_shapes	
:�*
dtype02&
$dense_Dense14/BiasAdd/ReadVariableOp�
dense_Dense14/BiasAddBiasAdddense_Dense14/MatMul:product:0,dense_Dense14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_Dense14/BiasAdd�
dense_Dense14/ReluReludense_Dense14/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_Dense14/Relu�
IdentityIdentity dense_Dense14/Relu:activations:0%^dense_Dense13/BiasAdd/ReadVariableOp$^dense_Dense13/MatMul/ReadVariableOp%^dense_Dense14/BiasAdd/ReadVariableOp$^dense_Dense14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2L
$dense_Dense13/BiasAdd/ReadVariableOp$dense_Dense13/BiasAdd/ReadVariableOp2J
#dense_Dense13/MatMul/ReadVariableOp#dense_Dense13/MatMul/ReadVariableOp2L
$dense_Dense14/BiasAdd/ReadVariableOp$dense_Dense14/BiasAdd/ReadVariableOp2J
#dense_Dense14/MatMul/ReadVariableOp#dense_Dense14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_121
dense_dense13_input>
:dense_dense13_statefulpartitionedcall_dense_dense13_kernel<
8dense_dense13_statefulpartitionedcall_dense_dense13_bias>
:dense_dense14_statefulpartitionedcall_dense_dense14_kernel<
8dense_dense14_statefulpartitionedcall_dense_dense14_bias
identity��%dense_Dense13/StatefulPartitionedCall�%dense_Dense14/StatefulPartitionedCall�
%dense_Dense13/StatefulPartitionedCallStatefulPartitionedCalldense_dense13_input:dense_dense13_statefulpartitionedcall_dense_dense13_kernel8dense_dense13_statefulpartitionedcall_dense_dense13_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_852'
%dense_Dense13/StatefulPartitionedCall�
%dense_Dense14/StatefulPartitionedCallStatefulPartitionedCall.dense_Dense13/StatefulPartitionedCall:output:0:dense_dense14_statefulpartitionedcall_dense_dense14_kernel8dense_dense14_statefulpartitionedcall_dense_dense14_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_1082'
%dense_Dense14/StatefulPartitionedCall�
IdentityIdentity.dense_Dense14/StatefulPartitionedCall:output:0&^dense_Dense13/StatefulPartitionedCall&^dense_Dense14/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2N
%dense_Dense13/StatefulPartitionedCall%dense_Dense13/StatefulPartitionedCall2N
%dense_Dense14/StatefulPartitionedCall%dense_Dense14/StatefulPartitionedCall:3 /
-
_user_specified_namedense_Dense13_input
�
�
*__inference_sequential_7_layer_call_fn_234

inputs0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_7_layer_call_and_return_conditional_losses_1632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
*__inference_sequential_7_layer_call_fn_170
dense_dense13_input0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_dense13_input,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_7_layer_call_and_return_conditional_losses_1632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:3 /
-
_user_specified_namedense_Dense13_input
�	
�
*__inference_sequential_7_layer_call_fn_151
dense_dense13_input0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_dense13_input,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_7_layer_call_and_return_conditional_losses_1442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:3 /
-
_user_specified_namedense_Dense13_input
�
�
*__inference_sequential_7_layer_call_fn_225

inputs0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_7_layer_call_and_return_conditional_losses_1442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_85

inputs.
*matmul_readvariableop_dense_dense13_kernel-
)biasadd_readvariableop_dense_dense13_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp*matmul_readvariableop_dense_dense13_kernel*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_dense_dense13_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
__inference__traced_save_306
file_prefix3
/savev2_dense_dense13_kernel_read_readvariableop1
-savev2_dense_dense13_bias_read_readvariableop3
/savev2_dense_dense14_kernel_read_readvariableop1
-savev2_dense_dense14_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3f775124d1544d658bbebbb6f407a66e/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_dense_dense13_kernel_read_readvariableop-savev2_dense_dense13_bias_read_readvariableop/savev2_dense_dense14_kernel_read_readvariableop-savev2_dense_dense14_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*:
_input_shapes)
': :	�@:@:	@�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
+__inference_dense_Dense13_layer_call_fn_252

inputs0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_108

inputs.
*matmul_readvariableop_dense_dense14_kernel-
)biasadd_readvariableop_dense_dense14_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp*matmul_readvariableop_dense_dense14_kernel*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_dense_dense14_bias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_163

inputs>
:dense_dense13_statefulpartitionedcall_dense_dense13_kernel<
8dense_dense13_statefulpartitionedcall_dense_dense13_bias>
:dense_dense14_statefulpartitionedcall_dense_dense14_kernel<
8dense_dense14_statefulpartitionedcall_dense_dense14_bias
identity��%dense_Dense13/StatefulPartitionedCall�%dense_Dense14/StatefulPartitionedCall�
%dense_Dense13/StatefulPartitionedCallStatefulPartitionedCallinputs:dense_dense13_statefulpartitionedcall_dense_dense13_kernel8dense_dense13_statefulpartitionedcall_dense_dense13_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_852'
%dense_Dense13/StatefulPartitionedCall�
%dense_Dense14/StatefulPartitionedCallStatefulPartitionedCall.dense_Dense13/StatefulPartitionedCall:output:0:dense_dense14_statefulpartitionedcall_dense_dense14_kernel8dense_dense14_statefulpartitionedcall_dense_dense14_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_1082'
%dense_Dense14/StatefulPartitionedCall�
IdentityIdentity.dense_Dense14/StatefulPartitionedCall:output:0&^dense_Dense13/StatefulPartitionedCall&^dense_Dense14/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2N
%dense_Dense13/StatefulPartitionedCall%dense_Dense13/StatefulPartitionedCall2N
%dense_Dense14/StatefulPartitionedCall%dense_Dense14/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_144

inputs>
:dense_dense13_statefulpartitionedcall_dense_dense13_kernel<
8dense_dense13_statefulpartitionedcall_dense_dense13_bias>
:dense_dense14_statefulpartitionedcall_dense_dense14_kernel<
8dense_dense14_statefulpartitionedcall_dense_dense14_bias
identity��%dense_Dense13/StatefulPartitionedCall�%dense_Dense14/StatefulPartitionedCall�
%dense_Dense13/StatefulPartitionedCallStatefulPartitionedCallinputs:dense_dense13_statefulpartitionedcall_dense_dense13_kernel8dense_dense13_statefulpartitionedcall_dense_dense13_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_852'
%dense_Dense13/StatefulPartitionedCall�
%dense_Dense14/StatefulPartitionedCallStatefulPartitionedCall.dense_Dense13/StatefulPartitionedCall:output:0:dense_dense14_statefulpartitionedcall_dense_dense14_kernel8dense_dense14_statefulpartitionedcall_dense_dense14_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_1082'
%dense_Dense14/StatefulPartitionedCall�
IdentityIdentity.dense_Dense14/StatefulPartitionedCall:output:0&^dense_Dense13/StatefulPartitionedCall&^dense_Dense14/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2N
%dense_Dense13/StatefulPartitionedCall%dense_Dense13/StatefulPartitionedCall2N
%dense_Dense14/StatefulPartitionedCall%dense_Dense14/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
E__inference_sequential_7_layer_call_and_return_conditional_losses_131
dense_dense13_input>
:dense_dense13_statefulpartitionedcall_dense_dense13_kernel<
8dense_dense13_statefulpartitionedcall_dense_dense13_bias>
:dense_dense14_statefulpartitionedcall_dense_dense14_kernel<
8dense_dense14_statefulpartitionedcall_dense_dense14_bias
identity��%dense_Dense13/StatefulPartitionedCall�%dense_Dense14/StatefulPartitionedCall�
%dense_Dense13/StatefulPartitionedCallStatefulPartitionedCalldense_dense13_input:dense_dense13_statefulpartitionedcall_dense_dense13_kernel8dense_dense13_statefulpartitionedcall_dense_dense13_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_Dense13_layer_call_and_return_conditional_losses_852'
%dense_Dense13/StatefulPartitionedCall�
%dense_Dense14/StatefulPartitionedCallStatefulPartitionedCall.dense_Dense13/StatefulPartitionedCall:output:0:dense_dense14_statefulpartitionedcall_dense_dense14_kernel8dense_dense14_statefulpartitionedcall_dense_dense14_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_1082'
%dense_Dense14/StatefulPartitionedCall�
IdentityIdentity.dense_Dense14/StatefulPartitionedCall:output:0&^dense_Dense13/StatefulPartitionedCall&^dense_Dense14/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2N
%dense_Dense13/StatefulPartitionedCall%dense_Dense13/StatefulPartitionedCall2N
%dense_Dense14/StatefulPartitionedCall%dense_Dense14/StatefulPartitionedCall:3 /
-
_user_specified_namedense_Dense13_input
�
�
!__inference_signature_wrapper_180
dense_dense13_input0
,statefulpartitionedcall_dense_dense13_kernel.
*statefulpartitionedcall_dense_dense13_bias0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_dense13_input,statefulpartitionedcall_dense_dense13_kernel*statefulpartitionedcall_dense_dense13_bias,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*&
f!R
__inference__wrapped_model_702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:3 /
-
_user_specified_namedense_Dense13_input
�
�
__inference__wrapped_model_70
dense_dense13_inputI
Esequential_7_dense_dense13_matmul_readvariableop_dense_dense13_kernelH
Dsequential_7_dense_dense13_biasadd_readvariableop_dense_dense13_biasI
Esequential_7_dense_dense14_matmul_readvariableop_dense_dense14_kernelH
Dsequential_7_dense_dense14_biasadd_readvariableop_dense_dense14_bias
identity��1sequential_7/dense_Dense13/BiasAdd/ReadVariableOp�0sequential_7/dense_Dense13/MatMul/ReadVariableOp�1sequential_7/dense_Dense14/BiasAdd/ReadVariableOp�0sequential_7/dense_Dense14/MatMul/ReadVariableOp�
0sequential_7/dense_Dense13/MatMul/ReadVariableOpReadVariableOpEsequential_7_dense_dense13_matmul_readvariableop_dense_dense13_kernel*
_output_shapes
:	�@*
dtype022
0sequential_7/dense_Dense13/MatMul/ReadVariableOp�
!sequential_7/dense_Dense13/MatMulMatMuldense_dense13_input8sequential_7/dense_Dense13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2#
!sequential_7/dense_Dense13/MatMul�
1sequential_7/dense_Dense13/BiasAdd/ReadVariableOpReadVariableOpDsequential_7_dense_dense13_biasadd_readvariableop_dense_dense13_bias*
_output_shapes
:@*
dtype023
1sequential_7/dense_Dense13/BiasAdd/ReadVariableOp�
"sequential_7/dense_Dense13/BiasAddBiasAdd+sequential_7/dense_Dense13/MatMul:product:09sequential_7/dense_Dense13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2$
"sequential_7/dense_Dense13/BiasAdd�
sequential_7/dense_Dense13/ReluRelu+sequential_7/dense_Dense13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2!
sequential_7/dense_Dense13/Relu�
0sequential_7/dense_Dense14/MatMul/ReadVariableOpReadVariableOpEsequential_7_dense_dense14_matmul_readvariableop_dense_dense14_kernel*
_output_shapes
:	@�*
dtype022
0sequential_7/dense_Dense14/MatMul/ReadVariableOp�
!sequential_7/dense_Dense14/MatMulMatMul-sequential_7/dense_Dense13/Relu:activations:08sequential_7/dense_Dense14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!sequential_7/dense_Dense14/MatMul�
1sequential_7/dense_Dense14/BiasAdd/ReadVariableOpReadVariableOpDsequential_7_dense_dense14_biasadd_readvariableop_dense_dense14_bias*
_output_shapes	
:�*
dtype023
1sequential_7/dense_Dense14/BiasAdd/ReadVariableOp�
"sequential_7/dense_Dense14/BiasAddBiasAdd+sequential_7/dense_Dense14/MatMul:product:09sequential_7/dense_Dense14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"sequential_7/dense_Dense14/BiasAdd�
sequential_7/dense_Dense14/ReluRelu+sequential_7/dense_Dense14/BiasAdd:output:0*
T0*(
_output_shapes
:����������2!
sequential_7/dense_Dense14/Relu�
IdentityIdentity-sequential_7/dense_Dense14/Relu:activations:02^sequential_7/dense_Dense13/BiasAdd/ReadVariableOp1^sequential_7/dense_Dense13/MatMul/ReadVariableOp2^sequential_7/dense_Dense14/BiasAdd/ReadVariableOp1^sequential_7/dense_Dense14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2f
1sequential_7/dense_Dense13/BiasAdd/ReadVariableOp1sequential_7/dense_Dense13/BiasAdd/ReadVariableOp2d
0sequential_7/dense_Dense13/MatMul/ReadVariableOp0sequential_7/dense_Dense13/MatMul/ReadVariableOp2f
1sequential_7/dense_Dense14/BiasAdd/ReadVariableOp1sequential_7/dense_Dense14/BiasAdd/ReadVariableOp2d
0sequential_7/dense_Dense14/MatMul/ReadVariableOp0sequential_7/dense_Dense14/MatMul/ReadVariableOp:3 /
-
_user_specified_namedense_Dense13_input
�
�
+__inference_dense_Dense14_layer_call_fn_270

inputs0
,statefulpartitionedcall_dense_dense14_kernel.
*statefulpartitionedcall_dense_dense14_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs,statefulpartitionedcall_dense_dense14_kernel*statefulpartitionedcall_dense_dense14_bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_1082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_263

inputs.
*matmul_readvariableop_dense_dense14_kernel-
)biasadd_readvariableop_dense_dense14_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp*matmul_readvariableop_dense_dense14_kernel*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_dense_dense14_bias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
__inference__traced_restore_330
file_prefix)
%assignvariableop_dense_dense13_kernel)
%assignvariableop_1_dense_dense13_bias+
'assignvariableop_2_dense_dense14_kernel)
%assignvariableop_3_dense_dense14_bias

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_dense_dense13_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp%assignvariableop_1_dense_dense13_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp'assignvariableop_2_dense_dense14_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp%assignvariableop_3_dense_dense14_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
T
dense_Dense13_input=
%serving_default_dense_Dense13_input:0����������B
dense_Dense141
StatefulPartitionedCall:0����������tensorflow/serving/predict:�Y
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
!_default_save_signature
*"&call_and_return_all_conditional_losses
#__call__"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_7", "layers": [{"class_name": "Dense", "config": {"name": "dense_Dense13", "trainable": true, "batch_input_shape": [null, 3072], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_Dense14", "trainable": true, "dtype": "float32", "units": 3072, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3072}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "Dense", "config": {"name": "dense_Dense13", "trainable": true, "batch_input_shape": [null, 3072], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_Dense14", "trainable": true, "dtype": "float32", "units": 3072, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "dense_Dense13_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 3072], "config": {"batch_input_shape": [null, 3072], "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_Dense13_input"}}
�

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
*$&call_and_return_all_conditional_losses
%__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_Dense13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 3072], "config": {"name": "dense_Dense13", "trainable": true, "batch_input_shape": [null, 3072], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3072}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*&&call_and_return_all_conditional_losses
'__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_Dense14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_Dense14", "trainable": true, "dtype": "float32", "units": 3072, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1, "mode": "fan_avg", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
layer_regularization_losses
trainable_variables
#__call__
!_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
,
(serving_default"
signature_map
':%	�@2dense_Dense13/kernel
 :@2dense_Dense13/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
layer_regularization_losses
trainable_variables
%__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
':%	@�2dense_Dense14/kernel
!:�2dense_Dense14/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
non_trainable_variables
metrics
regularization_losses

layers
 layer_regularization_losses
trainable_variables
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference__wrapped_model_70�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+
dense_Dense13_input����������
�2�
E__inference_sequential_7_layer_call_and_return_conditional_losses_198
E__inference_sequential_7_layer_call_and_return_conditional_losses_216
E__inference_sequential_7_layer_call_and_return_conditional_losses_131
E__inference_sequential_7_layer_call_and_return_conditional_losses_121�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_sequential_7_layer_call_fn_170
*__inference_sequential_7_layer_call_fn_225
*__inference_sequential_7_layer_call_fn_151
*__inference_sequential_7_layer_call_fn_234�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dense_Dense13_layer_call_and_return_conditional_losses_245�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_Dense13_layer_call_fn_252�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_263�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_Dense14_layer_call_fn_270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<B:
!__inference_signature_wrapper_180dense_Dense13_input�
__inference__wrapped_model_70�	
=�:
3�0
.�+
dense_Dense13_input����������
� ">�;
9
dense_Dense14(�%
dense_Dense14�����������
F__inference_dense_Dense13_layer_call_and_return_conditional_losses_245]	
0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� 
+__inference_dense_Dense13_layer_call_fn_252P	
0�-
&�#
!�
inputs����������
� "����������@�
F__inference_dense_Dense14_layer_call_and_return_conditional_losses_263]/�,
%�"
 �
inputs���������@
� "&�#
�
0����������
� 
+__inference_dense_Dense14_layer_call_fn_270P/�,
%�"
 �
inputs���������@
� "������������
E__inference_sequential_7_layer_call_and_return_conditional_losses_121u	
E�B
;�8
.�+
dense_Dense13_input����������
p

 
� "&�#
�
0����������
� �
E__inference_sequential_7_layer_call_and_return_conditional_losses_131u	
E�B
;�8
.�+
dense_Dense13_input����������
p 

 
� "&�#
�
0����������
� �
E__inference_sequential_7_layer_call_and_return_conditional_losses_198h	
8�5
.�+
!�
inputs����������
p

 
� "&�#
�
0����������
� �
E__inference_sequential_7_layer_call_and_return_conditional_losses_216h	
8�5
.�+
!�
inputs����������
p 

 
� "&�#
�
0����������
� �
*__inference_sequential_7_layer_call_fn_151h	
E�B
;�8
.�+
dense_Dense13_input����������
p

 
� "������������
*__inference_sequential_7_layer_call_fn_170h	
E�B
;�8
.�+
dense_Dense13_input����������
p 

 
� "������������
*__inference_sequential_7_layer_call_fn_225[	
8�5
.�+
!�
inputs����������
p

 
� "������������
*__inference_sequential_7_layer_call_fn_234[	
8�5
.�+
!�
inputs����������
p 

 
� "������������
!__inference_signature_wrapper_180�	
T�Q
� 
J�G
E
dense_Dense13_input.�+
dense_Dense13_input����������">�;
9
dense_Dense14(�%
dense_Dense14����������