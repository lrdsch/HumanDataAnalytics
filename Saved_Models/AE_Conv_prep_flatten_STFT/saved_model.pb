ч
Я
.
Abs
x"T
y"T"
Ttype:

2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758

Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v

2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
Є
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/v

4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
: *
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	  *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	  *
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m

2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
Є
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/m

4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
: *
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	  *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	  *
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0

conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel

-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
: *
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	  *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0

serving_default_input_6Placeholder*0
_output_shapes
:џџџџџџџџџ@*
dtype0*%
shape:џџџџџџџџџ@
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6conv2d_1/kernelconv2d_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasconv2d_transpose_1/kernelconv2d_transpose_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1354370

NoOpNoOp
ў\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Й\
valueЏ\BЌ\ BЅ\
Ь
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories*
Њ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories*
<
&0
'1
(2
)3
*4
+5
,6
-7*
<
&0
'1
(2
)3
*4
+5
,6
-7*
* 
А
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 
ф
;iter

<beta_1

=beta_2
	>decay
?learning_rate&mъ'mы(mь)mэ*mю+mя,m№-mё&vђ'vѓ(vє)vѕ*vі+vї,vј-vљ*

@serving_default* 
* 
* 
э
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

&kernel
'bias
#G_self_saveable_object_factories
 H_jit_compiled_convolution_op*
Г
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
#O_self_saveable_object_factories* 
Г
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
#V_self_saveable_object_factories* 
Ы
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

(kernel
)bias
#]_self_saveable_object_factories*
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 

^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
6
gtrace_0
htrace_1
itrace_2
jtrace_3* 
* 
Ы
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

*kernel
+bias
#q_self_saveable_object_factories*
Г
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
#x_self_saveable_object_factories* 
ю
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

,kernel
-bias
#_self_saveable_object_factories
!_jit_compiled_convolution_op*
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories* 
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories* 
 
*0
+1
,2
-3*
 
*0
+1
,2
-3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
OI
VARIABLE_VALUEconv2d_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

&0
'1*

&0
'1*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

Ѓtrace_0* 

Єtrace_0* 
* 
* 
* 
* 
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

Њtrace_0* 

Ћtrace_0* 
* 
* 
* 
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

Бtrace_0* 

Вtrace_0* 
* 

(0
)1*

(0
)1*
* 
З
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
Иactivity_regularizer_fn
*\&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
* 
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*

*0
+1*
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

Сtrace_0* 

Тtrace_0* 
* 
* 
* 
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

Шtrace_0* 

Щtrace_0* 
* 

,0
-1*

,0
-1*
* 

Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

Яtrace_0* 

аtrace_0* 
* 
* 
* 
* 
* 

бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

жtrace_0* 

зtrace_0* 
* 
* 
* 
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 
* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
п	variables
р	keras_api

сtotal

тcount*
M
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

шtrace_0* 

щtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

с0
т1*

п	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

х0
ц1*

у	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_2/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_2/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_3/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_2/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_2/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_3/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasconv2d_transpose_1/kernelconv2d_transpose_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/mAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/vConst*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1355191
Р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasconv2d_transpose_1/kernelconv2d_transpose_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/mAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1355300За
ћ#

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1354942

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
ж
)__inference_Decoder_layer_call_fn_1354677

inputs
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354066x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
Ш
D__inference_Encoder_layer_call_and_return_conditional_losses_1353739
input_4*
conv2d_1_1353698:
conv2d_1_1353700:"
dense_2_1353724:	  
dense_2_1353726: 
identity

identity_1Ђ conv2d_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallџ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_1_1353698conv2d_1_1353700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697ѕ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663с
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_1353724dense_2_1353726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723Ы
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *9
f4R2
0__inference_dense_2_activity_regularizer_1353682
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_4

ў
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ*w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ђ
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К
ж
)__inference_Encoder_layer_call_fn_1353843
input_4!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_4


н
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354414

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354277x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


"__inference__wrapped_model_1353657
input_6c
Iae_conv_prep_flatten_stft_encoder_conv2d_1_conv2d_readvariableop_resource:X
Jae_conv_prep_flatten_stft_encoder_conv2d_1_biasadd_readvariableop_resource:[
Hae_conv_prep_flatten_stft_encoder_dense_2_matmul_readvariableop_resource:	  W
Iae_conv_prep_flatten_stft_encoder_dense_2_biasadd_readvariableop_resource: [
Hae_conv_prep_flatten_stft_decoder_dense_3_matmul_readvariableop_resource:	  X
Iae_conv_prep_flatten_stft_decoder_dense_3_biasadd_readvariableop_resource:	 w
]ae_conv_prep_flatten_stft_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:b
Tae_conv_prep_flatten_stft_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:
identityЂKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpЂTAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOpЂ?AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOpЂAAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOpЂ?AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOpв
@AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ё
1AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2DConv2Dinput_6HAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ш
AAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_stft_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
2AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAddBiasAdd:AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D:output:0IAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*Ќ
.AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/EluElu;AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*ю
9AE_Conv_prep_flatten_STFT/Encoder/max_pooling2d_1/MaxPoolMaxPool<AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

1AE_Conv_prep_flatten_STFT/Encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ё
3AE_Conv_prep_flatten_STFT/Encoder/flatten_1/ReshapeReshapeBAE_Conv_prep_flatten_STFT/Encoder/max_pooling2d_1/MaxPool:output:0:AE_Conv_prep_flatten_STFT/Encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Щ
?AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOpReadVariableOpHae_conv_prep_flatten_stft_encoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0ѓ
0AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMulMatMul<AE_Conv_prep_flatten_STFT/Encoder/flatten_1/Reshape:output:0GAE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ц
@AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
1AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAddBiasAdd:AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul:product:0HAE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
-AE_Conv_prep_flatten_STFT/Encoder/dense_2/EluElu:AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ З
AAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/AbsAbs;AE_Conv_prep_flatten_STFT/Encoder/dense_2/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
CAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ў
AAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/SumSumEAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/Abs:y:0LAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 
CAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
AAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/mulMulLAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/mul/x:output:0JAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: М
CAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/ShapeShape;AE_Conv_prep_flatten_STFT/Encoder/dense_2/Elu:activations:0*
T0*
_output_shapes
::эЯ
QAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
SAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
SAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
KAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_sliceStridedSliceLAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/Shape:output:0ZAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stack:output:0\AE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0\AE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskа
BAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/CastCastTAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
EAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/truedivRealDivEAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/mul:z:0FAE_Conv_prep_flatten_STFT/Encoder/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Щ
?AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOpReadVariableOpHae_conv_prep_flatten_stft_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0ѓ
0AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMulMatMul;AE_Conv_prep_flatten_STFT/Encoder/dense_2/Elu:activations:0GAE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ Ч
@AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0ѕ
1AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAddBiasAdd:AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul:product:0HAE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ Ѓ
-AE_Conv_prep_flatten_STFT/Decoder/dense_3/EluElu:AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Њ
1AE_Conv_prep_flatten_STFT/Decoder/reshape_1/ShapeShape;AE_Conv_prep_flatten_STFT/Decoder/dense_3/Elu:activations:0*
T0*
_output_shapes
::эЯ
?AE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9AE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_sliceStridedSlice:AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Shape:output:0HAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stack:output:0JAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stack_1:output:0JAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
9AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shapePackBAE_Conv_prep_flatten_STFT/Decoder/reshape_1/strided_slice:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/1:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/2:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:љ
3AE_Conv_prep_flatten_STFT/Decoder/reshape_1/ReshapeReshape;AE_Conv_prep_flatten_STFT/Decoder/dense_3/Elu:activations:0BAE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџД
:AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/ShapeShape<AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
::эЯ
HAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
BAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_sliceStridedSliceCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/Shape:output:0QAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stack:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stack_1:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
:AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stackPackKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/1:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/2:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
LAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
LAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
DAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1StridedSliceCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stack:output:0UAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0UAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
TAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp]ae_conv_prep_flatten_stft_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0 
EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInputCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/stack:output:0\AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0<AE_Conv_prep_flatten_STFT/Decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
м
KAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_stft_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAddBiasAddNAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*Р
8AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/EluEluEAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
7AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   
9AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ч
5AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/mulMul@AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/Const:output:0BAE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:О
NAE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborFAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/Elu:activations:09AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(
8AE_Conv_prep_flatten_STFT/Decoder/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      Э
BAE_Conv_prep_flatten_STFT/Decoder/resizing_1/resize/ResizeBilinearResizeBilinear_AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0AAE_Conv_prep_flatten_STFT/Decoder/resizing_1/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(Ћ
IdentityIdentitySAE_Conv_prep_flatten_STFT/Decoder/resizing_1/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ќ
NoOpNoOpL^AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpU^AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOp@^AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOpB^AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOp@^AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2
KAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2Ќ
TAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpTAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOp@AE_Conv_prep_flatten_STFT/Decoder/dense_3/BiasAdd/ReadVariableOp2
?AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOp?AE_Conv_prep_flatten_STFT/Decoder/dense_3/MatMul/ReadVariableOp2
AAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_STFT/Encoder/conv2d_1/BiasAdd/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOp@AE_Conv_prep_flatten_STFT/Encoder/conv2d_1/Conv2D/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOp@AE_Conv_prep_flatten_STFT/Encoder/dense_2/BiasAdd/ReadVariableOp2
?AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOp?AE_Conv_prep_flatten_STFT/Encoder/dense_2/MatMul/ReadVariableOp:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6
ў
ж
D__inference_Decoder_layer_call_and_return_conditional_losses_1354066

inputs"
dense_3_1354052:	  
dense_3_1354054:	 4
conv2d_transpose_1_1354058:(
conv2d_transpose_1_1354060:
identityЂ*conv2d_transpose_1/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallѓ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_1354052dense_3_1354054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992ш
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012Т
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_1_1354058conv2d_transpose_1_1354060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971ы
resizing_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026{
IdentityIdentity#resizing_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


н
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354392

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354231x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
з
)__inference_Decoder_layer_call_fn_1354077
input_5
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354066x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_5

з
D__inference_Decoder_layer_call_and_return_conditional_losses_1354046
input_5"
dense_3_1354032:	  
dense_3_1354034:	 4
conv2d_transpose_1_1354038:(
conv2d_transpose_1_1354040:
identityЂ*conv2d_transpose_1/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_3_1354032dense_3_1354034*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992ш
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012Т
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_1_1354038conv2d_transpose_1_1354040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971ы
resizing_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026{
IdentityIdentity#resizing_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_5
 

ї
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992

inputs1
matmul_readvariableop_resource:	  .
biasadd_readvariableop_resource:	 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
д
ф
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354231

inputs)
encoder_1354210:
encoder_1354212:"
encoder_1354214:	  
encoder_1354216: "
decoder_1354220:	  
decoder_1354222:	 )
decoder_1354224:
decoder_1354226:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_1354210encoder_1354212encoder_1354214encoder_1354216*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353792У
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_1354220decoder_1354222decoder_1354224decoder_1354226*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354066
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
хё

 __inference__traced_save_1355191
file_prefix@
&read_disablecopyonread_conv2d_1_kernel:4
&read_1_disablecopyonread_conv2d_1_bias::
'read_2_disablecopyonread_dense_2_kernel:	  3
%read_3_disablecopyonread_dense_2_bias: :
'read_4_disablecopyonread_dense_3_kernel:	  4
%read_5_disablecopyonread_dense_3_bias:	 L
2read_6_disablecopyonread_conv2d_transpose_1_kernel:>
0read_7_disablecopyonread_conv2d_transpose_1_bias:,
"read_8_disablecopyonread_adam_iter:	 .
$read_9_disablecopyonread_adam_beta_1: /
%read_10_disablecopyonread_adam_beta_2: .
$read_11_disablecopyonread_adam_decay: 6
,read_12_disablecopyonread_adam_learning_rate: +
!read_13_disablecopyonread_total_1: +
!read_14_disablecopyonread_count_1: )
read_15_disablecopyonread_total: )
read_16_disablecopyonread_count: J
0read_17_disablecopyonread_adam_conv2d_1_kernel_m:<
.read_18_disablecopyonread_adam_conv2d_1_bias_m:B
/read_19_disablecopyonread_adam_dense_2_kernel_m:	  ;
-read_20_disablecopyonread_adam_dense_2_bias_m: B
/read_21_disablecopyonread_adam_dense_3_kernel_m:	  <
-read_22_disablecopyonread_adam_dense_3_bias_m:	 T
:read_23_disablecopyonread_adam_conv2d_transpose_1_kernel_m:F
8read_24_disablecopyonread_adam_conv2d_transpose_1_bias_m:J
0read_25_disablecopyonread_adam_conv2d_1_kernel_v:<
.read_26_disablecopyonread_adam_conv2d_1_bias_v:B
/read_27_disablecopyonread_adam_dense_2_kernel_v:	  ;
-read_28_disablecopyonread_adam_dense_2_bias_v: B
/read_29_disablecopyonread_adam_dense_3_kernel_v:	  <
-read_30_disablecopyonread_adam_dense_3_bias_v:	 T
:read_31_disablecopyonread_adam_conv2d_transpose_1_kernel_v:F
8read_32_disablecopyonread_adam_conv2d_transpose_1_bias_v:
savev2_const
identity_67ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 Њ
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	  y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 Ё
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_3_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	  y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_3_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
: *
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
: b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
: 
Read_6/DisableCopyOnReadDisableCopyOnRead2read_6_disablecopyonread_conv2d_transpose_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_6/ReadVariableOpReadVariableOp2read_6_disablecopyonread_conv2d_transpose_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_conv2d_transpose_1_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_conv2d_transpose_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_adam_iter^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: x
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_adam_beta_1^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_adam_beta_2^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_11/DisableCopyOnReadDisableCopyOnRead$read_11_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 
Read_11/ReadVariableOpReadVariableOp$read_11_disablecopyonread_adam_decay^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_12/DisableCopyOnReadDisableCopyOnRead,read_12_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 І
Read_12/ReadVariableOpReadVariableOp,read_12_disablecopyonread_adam_learning_rate^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_13/DisableCopyOnReadDisableCopyOnRead!read_13_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_13/ReadVariableOpReadVariableOp!read_13_disablecopyonread_total_1^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_14/DisableCopyOnReadDisableCopyOnRead!read_14_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_14/ReadVariableOpReadVariableOp!read_14_disablecopyonread_count_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_15/DisableCopyOnReadDisableCopyOnReadread_15_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_15/ReadVariableOpReadVariableOpread_15_disablecopyonread_total^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_count^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_conv2d_1_kernel_m"/device:CPU:0*
_output_shapes
 К
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_conv2d_1_kernel_m^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_conv2d_1_bias_m"/device:CPU:0*
_output_shapes
 Ќ
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_conv2d_1_bias_m^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_dense_2_kernel_m"/device:CPU:0*
_output_shapes
 В
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_dense_2_kernel_m^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	  
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_dense_2_bias_m"/device:CPU:0*
_output_shapes
 Ћ
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_dense_2_bias_m^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_dense_3_kernel_m"/device:CPU:0*
_output_shapes
 В
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_dense_3_kernel_m^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0p
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  f
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	  
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_adam_dense_3_bias_m"/device:CPU:0*
_output_shapes
 Ќ
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_adam_dense_3_bias_m^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
: *
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
: b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
: 
Read_23/DisableCopyOnReadDisableCopyOnRead:read_23_disablecopyonread_adam_conv2d_transpose_1_kernel_m"/device:CPU:0*
_output_shapes
 Ф
Read_23/ReadVariableOpReadVariableOp:read_23_disablecopyonread_adam_conv2d_transpose_1_kernel_m^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_24/DisableCopyOnReadDisableCopyOnRead8read_24_disablecopyonread_adam_conv2d_transpose_1_bias_m"/device:CPU:0*
_output_shapes
 Ж
Read_24/ReadVariableOpReadVariableOp8read_24_disablecopyonread_adam_conv2d_transpose_1_bias_m^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_conv2d_1_kernel_v"/device:CPU:0*
_output_shapes
 К
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_conv2d_1_kernel_v^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_conv2d_1_bias_v"/device:CPU:0*
_output_shapes
 Ќ
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_conv2d_1_bias_v^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_dense_2_kernel_v"/device:CPU:0*
_output_shapes
 В
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_dense_2_kernel_v^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	  
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_dense_2_bias_v"/device:CPU:0*
_output_shapes
 Ћ
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_dense_2_bias_v^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_adam_dense_3_kernel_v"/device:CPU:0*
_output_shapes
 В
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_adam_dense_3_kernel_v^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	  *
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	  f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	  
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_dense_3_bias_v"/device:CPU:0*
_output_shapes
 Ќ
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_dense_3_bias_v^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
: *
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
: b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
: 
Read_31/DisableCopyOnReadDisableCopyOnRead:read_31_disablecopyonread_adam_conv2d_transpose_1_kernel_v"/device:CPU:0*
_output_shapes
 Ф
Read_31/ReadVariableOpReadVariableOp:read_31_disablecopyonread_adam_conv2d_transpose_1_kernel_v^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_32/DisableCopyOnReadDisableCopyOnRead8read_32_disablecopyonread_adam_conv2d_transpose_1_bias_v"/device:CPU:0*
_output_shapes
 Ж
Read_32/ReadVariableOpReadVariableOp8read_32_disablecopyonread_adam_conv2d_transpose_1_bias_v^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:Ч
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*№
valueцBу"B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_66Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_67IdentityIdentity_66:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_67Identity_67:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:"

_output_shapes
: 
 

ї
D__inference_dense_3_layer_call_and_return_conditional_losses_1354876

inputs1
matmul_readvariableop_resource:	  .
biasadd_readvariableop_resource:	 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
g
Є
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354568

inputsI
/encoder_conv2d_1_conv2d_readvariableop_resource:>
0encoder_conv2d_1_biasadd_readvariableop_resource:A
.encoder_dense_2_matmul_readvariableop_resource:	  =
/encoder_dense_2_biasadd_readvariableop_resource: A
.decoder_dense_3_matmul_readvariableop_resource:	  >
/decoder_dense_3_biasadd_readvariableop_resource:	 ]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:
identity

identity_1Ђ1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpЂ:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЂ&Decoder/dense_3/BiasAdd/ReadVariableOpЂ%Decoder/dense_3/MatMul/ReadVariableOpЂ'Encoder/conv2d_1/BiasAdd/ReadVariableOpЂ&Encoder/conv2d_1/Conv2D/ReadVariableOpЂ&Encoder/dense_2/BiasAdd/ReadVariableOpЂ%Encoder/dense_2/MatMul/ReadVariableOp
&Encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0М
Encoder/conv2d_1/Conv2DConv2Dinputs.Encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

'Encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Encoder/conv2d_1/BiasAddBiasAdd Encoder/conv2d_1/Conv2D:output:0/Encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*x
Encoder/conv2d_1/EluElu!Encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*К
Encoder/max_pooling2d_1/MaxPoolMaxPool"Encoder/conv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ѓ
Encoder/flatten_1/ReshapeReshape(Encoder/max_pooling2d_1/MaxPool:output:0 Encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
%Encoder/dense_2/MatMul/ReadVariableOpReadVariableOp.encoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ѕ
Encoder/dense_2/MatMulMatMul"Encoder/flatten_1/Reshape:output:0-Encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&Encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
Encoder/dense_2/BiasAddBiasAdd Encoder/dense_2/MatMul:product:0.Encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ n
Encoder/dense_2/EluElu Encoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'Encoder/dense_2/ActivityRegularizer/AbsAbs!Encoder/dense_2/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ z
)Encoder/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       А
'Encoder/dense_2/ActivityRegularizer/SumSum+Encoder/dense_2/ActivityRegularizer/Abs:y:02Encoder/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8Е
'Encoder/dense_2/ActivityRegularizer/mulMul2Encoder/dense_2/ActivityRegularizer/mul/x:output:00Encoder/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
)Encoder/dense_2/ActivityRegularizer/ShapeShape!Encoder/dense_2/Elu:activations:0*
T0*
_output_shapes
::эЯ
7Encoder/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9Encoder/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9Encoder/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1Encoder/dense_2/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_2/ActivityRegularizer/Shape:output:0@Encoder/dense_2/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(Encoder/dense_2/ActivityRegularizer/CastCast:Encoder/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: В
+Encoder/dense_2/ActivityRegularizer/truedivRealDiv+Encoder/dense_2/ActivityRegularizer/mul:z:0,Encoder/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
%Decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ѕ
Decoder/dense_3/MatMulMatMul!Encoder/dense_2/Elu:activations:0-Decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&Decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ї
Decoder/dense_3/BiasAddBiasAdd Decoder/dense_3/MatMul:product:0.Decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ o
Decoder/dense_3/EluElu Decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ v
Decoder/reshape_1/ShapeShape!Decoder/dense_3/Elu:activations:0*
T0*
_output_shapes
::эЯo
%Decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
Decoder/reshape_1/strided_sliceStridedSlice Decoder/reshape_1/Shape:output:0.Decoder/reshape_1/strided_slice/stack:output:00Decoder/reshape_1/strided_slice/stack_1:output:00Decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Decoder/reshape_1/Reshape/shapePack(Decoder/reshape_1/strided_slice:output:0*Decoder/reshape_1/Reshape/shape/1:output:0*Decoder/reshape_1/Reshape/shape/2:output:0*Decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ћ
Decoder/reshape_1/ReshapeReshape!Decoder/dense_3/Elu:activations:0(Decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 Decoder/conv2d_transpose_1/ShapeShape"Decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
::эЯx
.Decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(Decoder/conv2d_transpose_1/strided_sliceStridedSlice)Decoder/conv2d_transpose_1/Shape:output:07Decoder/conv2d_transpose_1/strided_slice/stack:output:09Decoder/conv2d_transpose_1/strided_slice/stack_1:output:09Decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*d
"Decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 Decoder/conv2d_transpose_1/stackPack1Decoder/conv2d_transpose_1/strided_slice:output:0+Decoder/conv2d_transpose_1/stack/1:output:0+Decoder/conv2d_transpose_1/stack/2:output:0+Decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*Decoder/conv2d_transpose_1/strided_slice_1StridedSlice)Decoder/conv2d_transpose_1/stack:output:09Decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+Decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_1/stack:output:0BDecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ј
1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
"Decoder/conv2d_transpose_1/BiasAddBiasAdd4Decoder/conv2d_transpose_1/conv2d_transpose:output:09Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Decoder/conv2d_transpose_1/EluElu+Decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*n
Decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   p
Decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_1/mulMul&Decoder/up_sampling2d_1/Const:output:0(Decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:№
4Decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_1/Elu:activations:0Decoder/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(o
Decoder/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      џ
(Decoder/resizing_1/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_1/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity9Decoder/resizing_1/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@o

Identity_1Identity/Encoder/dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ќ
NoOpNoOp2^Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp'^Decoder/dense_3/BiasAdd/ReadVariableOp&^Decoder/dense_3/MatMul/ReadVariableOp(^Encoder/conv2d_1/BiasAdd/ReadVariableOp'^Encoder/conv2d_1/Conv2D/ReadVariableOp'^Encoder/dense_2/BiasAdd/ReadVariableOp&^Encoder/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2f
1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_3/BiasAdd/ReadVariableOp&Decoder/dense_3/BiasAdd/ReadVariableOp2N
%Decoder/dense_3/MatMul/ReadVariableOp%Decoder/dense_3/MatMul/ReadVariableOp2R
'Encoder/conv2d_1/BiasAdd/ReadVariableOp'Encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_1/Conv2D/ReadVariableOp&Encoder/conv2d_1/Conv2D/ReadVariableOp2P
&Encoder/dense_2/BiasAdd/ReadVariableOp&Encoder/dense_2/BiasAdd/ReadVariableOp2N
%Encoder/dense_2/MatMul/ReadVariableOp%Encoder/dense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь
H
,__inference_resizing_1_layer_call_fn_1354964

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
ж
D__inference_Decoder_layer_call_and_return_conditional_losses_1354096

inputs"
dense_3_1354082:	  
dense_3_1354084:	 4
conv2d_transpose_1_1354088:(
conv2d_transpose_1_1354090:
identityЂ*conv2d_transpose_1/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallѓ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_1354082dense_3_1354084*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992ш
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012Т
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_1_1354088conv2d_transpose_1_1354090*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971ы
resizing_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026{
IdentityIdentity#resizing_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ц
H__inference_dense_2_layer_call_and_return_all_conditional_losses_1354845

inputs
unknown:	  
	unknown_0: 
identity

identity_1ЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723Ї
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *9
f4R2
0__inference_dense_2_activity_regularizer_1353682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Е
ж
)__inference_Decoder_layer_call_fn_1354690

inputs
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354096x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Д
#__inference__traced_restore_1355300
file_prefix:
 assignvariableop_conv2d_1_kernel:.
 assignvariableop_1_conv2d_1_bias:4
!assignvariableop_2_dense_2_kernel:	  -
assignvariableop_3_dense_2_bias: 4
!assignvariableop_4_dense_3_kernel:	  .
assignvariableop_5_dense_3_bias:	 F
,assignvariableop_6_conv2d_transpose_1_kernel:8
*assignvariableop_7_conv2d_transpose_1_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: D
*assignvariableop_17_adam_conv2d_1_kernel_m:6
(assignvariableop_18_adam_conv2d_1_bias_m:<
)assignvariableop_19_adam_dense_2_kernel_m:	  5
'assignvariableop_20_adam_dense_2_bias_m: <
)assignvariableop_21_adam_dense_3_kernel_m:	  6
'assignvariableop_22_adam_dense_3_bias_m:	 N
4assignvariableop_23_adam_conv2d_transpose_1_kernel_m:@
2assignvariableop_24_adam_conv2d_transpose_1_bias_m:D
*assignvariableop_25_adam_conv2d_1_kernel_v:6
(assignvariableop_26_adam_conv2d_1_bias_v:<
)assignvariableop_27_adam_dense_2_kernel_v:	  5
'assignvariableop_28_adam_dense_2_bias_v: <
)assignvariableop_29_adam_dense_3_kernel_v:	  6
'assignvariableop_30_adam_dense_3_bias_v:	 N
4assignvariableop_31_adam_conv2d_transpose_1_kernel_v:@
2assignvariableop_32_adam_conv2d_transpose_1_bias_v:
identity_34ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*№
valueцBу"B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_3_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_3_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_1_bias_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_conv2d_transpose_1_kernel_mIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_conv2d_transpose_1_bias_mIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_vIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_2_kernel_vIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_2_bias_vIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_3_kernel_vIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_3_bias_vIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_conv2d_transpose_1_kernel_vIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_conv2d_transpose_1_bias_vIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ѕ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
'
ф
D__inference_Encoder_layer_call_and_return_conditional_losses_1354664

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	  5
'dense_2_biasadd_readvariableop_resource: 
identity

identity_1Ђconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ќ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Њ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_1/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ^
dense_2/EluEludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ s
dense_2/ActivityRegularizer/AbsAbsdense_2/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ r
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: x
!dense_2/ActivityRegularizer/ShapeShapedense_2/Elu:activations:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Elu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ъ
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
Ш
D__inference_Encoder_layer_call_and_return_conditional_losses_1353764
input_4*
conv2d_1_1353742:
conv2d_1_1353744:"
dense_2_1353749:	  
dense_2_1353751: 
identity

identity_1Ђ conv2d_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallџ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_1_1353742conv2d_1_1353744*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697ѕ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663с
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_1353749dense_2_1353751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723Ы
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *9
f4R2
0__inference_dense_2_activity_regularizer_1353682
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_4
з
х
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354204
input_6)
encoder_1354183:
encoder_1354185:"
encoder_1354187:	  
encoder_1354189: "
decoder_1354193:	  
decoder_1354195:	 )
decoder_1354197:
decoder_1354199:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_6encoder_1354183encoder_1354185encoder_1354187encoder_1354189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353831У
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_1354193decoder_1354195decoder_1354197decoder_1354199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354096
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6
Ш

)__inference_dense_2_layer_call_fn_1354834

inputs
unknown:	  
	unknown_0: 
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Н
M
1__inference_max_pooling2d_1_layer_call_fn_1354809

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1354959

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:Е
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

з
D__inference_Decoder_layer_call_and_return_conditional_losses_1354029
input_5"
dense_3_1353993:	  
dense_3_1353995:	 4
conv2d_transpose_1_1354014:(
conv2d_transpose_1_1354016:
identityЂ*conv2d_transpose_1/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_3_1353993dense_3_1353995*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992ш
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012Т
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_1_1354014conv2d_transpose_1_1354016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971ы
resizing_1/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026{
IdentityIdentity#resizing_1/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_5
Щ

)__inference_dense_3_layer_call_fn_1354865

inputs
unknown:	  
	unknown_0:	 
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_1353992p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
Ч
D__inference_Encoder_layer_call_and_return_conditional_losses_1353792

inputs*
conv2d_1_1353770:
conv2d_1_1353772:"
dense_2_1353777:	  
dense_2_1353779: 
identity

identity_1Ђ conv2d_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_1353770conv2d_1_1353772*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697ѕ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663с
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_1353777dense_2_1353779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723Ы
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *9
f4R2
0__inference_dense_2_activity_regularizer_1353682
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ђ

о
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354251
input_6!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354231x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6
н
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354895

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ :P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
§
G
0__inference_dense_2_activity_regularizer_1353682
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџD
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
И
з
)__inference_Decoder_layer_call_fn_1354107
input_5
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354096x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_5
Н
M
1__inference_up_sampling2d_1_layer_call_fn_1354947

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1353971
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ё

*__inference_conv2d_1_layer_call_fn_1354793

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ж
G
+__inference_reshape_1_layer_call_fn_1354881

inputs
identityМ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ :P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
н
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354012

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ :P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ж
G
+__inference_flatten_1_layer_call_fn_1354819

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1354825

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё8

D__inference_Decoder_layer_call_and_return_conditional_losses_1354784

inputs9
&dense_3_matmul_readvariableop_resource:	  6
'dense_3_biasadd_readvariableop_resource:	 U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identityЂ)conv2d_transpose_1/BiasAdd/ReadVariableOpЂ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOp
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0z
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ _
dense_3/EluEludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ f
reshape_1/ShapeShapedense_3/Elu:activations:0*
T0*
_output_shapes
::эЯg
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapedense_3/Elu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
conv2d_transpose_1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
::эЯp
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*|
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:и
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_1/Elu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(g
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      ч
 resizing_1/resize/ResizeBilinearResizeBilinear=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0resizing_1/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity1resizing_1/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ш
NoOpNoOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
Ч
D__inference_Encoder_layer_call_and_return_conditional_losses_1353831

inputs*
conv2d_1_1353809:
conv2d_1_1353811:"
dense_2_1353816:	  
dense_2_1353818: 
identity

identity_1Ђ conv2d_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_1353809conv2d_1_1353811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1353697ѕ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663с
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_1353710
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_1353816dense_2_1353818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723Ы
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *9
f4R2
0__inference_dense_2_activity_regularizer_1353682
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


і
D__inference_dense_2_layer_call_and_return_conditional_losses_1354856

inputs1
matmul_readvariableop_resource:	  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ћ#

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
c
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354970

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


і
D__inference_dense_2_layer_call_and_return_conditional_losses_1353723

inputs1
matmul_readvariableop_resource:	  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
е	
Ш
%__inference_signature_wrapper_1354370
input_6!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1353657x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6

ў
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1354804

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ*w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
ж
)__inference_Encoder_layer_call_fn_1353804
input_4!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353792o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_4
Ё8

D__inference_Decoder_layer_call_and_return_conditional_losses_1354737

inputs9
&dense_3_matmul_readvariableop_resource:	  6
'dense_3_biasadd_readvariableop_resource:	 U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identityЂ)conv2d_transpose_1/BiasAdd/ReadVariableOpЂ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOp
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0z
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ _
dense_3/EluEludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ f
reshape_1/ShapeShapedense_3/Elu:activations:0*
T0*
_output_shapes
::эЯg
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapedense_3/Elu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
conv2d_transpose_1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
::эЯp
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*|
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:и
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_1/Elu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(g
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      ч
 resizing_1/resize/ResizeBilinearResizeBilinear=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0resizing_1/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity1resizing_1/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ш
NoOpNoOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Я
c
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354026

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(w
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
З
е
)__inference_Encoder_layer_call_fn_1354596

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д
ф
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354277

inputs)
encoder_1354256:
encoder_1354258:"
encoder_1354260:	  
encoder_1354262: "
decoder_1354266:	  
decoder_1354268:	 )
decoder_1354270:
decoder_1354272:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_1354256encoder_1354258encoder_1354260encoder_1354262*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353831У
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_1354266decoder_1354268decoder_1354270decoder_1354272*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354096
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
'
ф
D__inference_Encoder_layer_call_and_return_conditional_losses_1354630

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	  5
'dense_2_biasadd_readvariableop_resource: 
identity

identity_1Ђconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ќ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Њ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_1/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ^
dense_2/EluEludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ s
dense_2/ActivityRegularizer/AbsAbsdense_2/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ r
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: x
!dense_2/ActivityRegularizer/ShapeShapedense_2/Elu:activations:0*
T0*
_output_shapes
::эЯy
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Elu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ g

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ъ
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ь
Љ
4__inference_conv2d_transpose_1_layer_call_fn_1354904

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1353948
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1353663

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
з
х
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354180
input_6)
encoder_1354159:
encoder_1354161:"
encoder_1354163:	  
encoder_1354165: "
decoder_1354169:	  
decoder_1354171:	 )
decoder_1354173:
decoder_1354175:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_6encoder_1354159encoder_1354161encoder_1354163encoder_1354165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353792У
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_1354169decoder_1354171decoder_1354173decoder_1354175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_1354066
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@h

Identity_1Identity(Encoder/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6
g
Є
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354491

inputsI
/encoder_conv2d_1_conv2d_readvariableop_resource:>
0encoder_conv2d_1_biasadd_readvariableop_resource:A
.encoder_dense_2_matmul_readvariableop_resource:	  =
/encoder_dense_2_biasadd_readvariableop_resource: A
.decoder_dense_3_matmul_readvariableop_resource:	  >
/decoder_dense_3_biasadd_readvariableop_resource:	 ]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:
identity

identity_1Ђ1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpЂ:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЂ&Decoder/dense_3/BiasAdd/ReadVariableOpЂ%Decoder/dense_3/MatMul/ReadVariableOpЂ'Encoder/conv2d_1/BiasAdd/ReadVariableOpЂ&Encoder/conv2d_1/Conv2D/ReadVariableOpЂ&Encoder/dense_2/BiasAdd/ReadVariableOpЂ%Encoder/dense_2/MatMul/ReadVariableOp
&Encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0М
Encoder/conv2d_1/Conv2DConv2Dinputs.Encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

'Encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Encoder/conv2d_1/BiasAddBiasAdd Encoder/conv2d_1/Conv2D:output:0/Encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*x
Encoder/conv2d_1/EluElu!Encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*К
Encoder/max_pooling2d_1/MaxPoolMaxPool"Encoder/conv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ѓ
Encoder/flatten_1/ReshapeReshape(Encoder/max_pooling2d_1/MaxPool:output:0 Encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
%Encoder/dense_2/MatMul/ReadVariableOpReadVariableOp.encoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ѕ
Encoder/dense_2/MatMulMatMul"Encoder/flatten_1/Reshape:output:0-Encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&Encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
Encoder/dense_2/BiasAddBiasAdd Encoder/dense_2/MatMul:product:0.Encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ n
Encoder/dense_2/EluElu Encoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'Encoder/dense_2/ActivityRegularizer/AbsAbs!Encoder/dense_2/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ z
)Encoder/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       А
'Encoder/dense_2/ActivityRegularizer/SumSum+Encoder/dense_2/ActivityRegularizer/Abs:y:02Encoder/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: n
)Encoder/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8Е
'Encoder/dense_2/ActivityRegularizer/mulMul2Encoder/dense_2/ActivityRegularizer/mul/x:output:00Encoder/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
)Encoder/dense_2/ActivityRegularizer/ShapeShape!Encoder/dense_2/Elu:activations:0*
T0*
_output_shapes
::эЯ
7Encoder/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9Encoder/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9Encoder/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1Encoder/dense_2/ActivityRegularizer/strided_sliceStridedSlice2Encoder/dense_2/ActivityRegularizer/Shape:output:0@Encoder/dense_2/ActivityRegularizer/strided_slice/stack:output:0BEncoder/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0BEncoder/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(Encoder/dense_2/ActivityRegularizer/CastCast:Encoder/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: В
+Encoder/dense_2/ActivityRegularizer/truedivRealDiv+Encoder/dense_2/ActivityRegularizer/mul:z:0,Encoder/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
%Decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ѕ
Decoder/dense_3/MatMulMatMul!Encoder/dense_2/Elu:activations:0-Decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&Decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Ї
Decoder/dense_3/BiasAddBiasAdd Decoder/dense_3/MatMul:product:0.Decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ o
Decoder/dense_3/EluElu Decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ v
Decoder/reshape_1/ShapeShape!Decoder/dense_3/Elu:activations:0*
T0*
_output_shapes
::эЯo
%Decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
Decoder/reshape_1/strided_sliceStridedSlice Decoder/reshape_1/Shape:output:0.Decoder/reshape_1/strided_slice/stack:output:00Decoder/reshape_1/strided_slice/stack_1:output:00Decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Decoder/reshape_1/Reshape/shapePack(Decoder/reshape_1/strided_slice:output:0*Decoder/reshape_1/Reshape/shape/1:output:0*Decoder/reshape_1/Reshape/shape/2:output:0*Decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ћ
Decoder/reshape_1/ReshapeReshape!Decoder/dense_3/Elu:activations:0(Decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 Decoder/conv2d_transpose_1/ShapeShape"Decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
::эЯx
.Decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(Decoder/conv2d_transpose_1/strided_sliceStridedSlice)Decoder/conv2d_transpose_1/Shape:output:07Decoder/conv2d_transpose_1/strided_slice/stack:output:09Decoder/conv2d_transpose_1/strided_slice/stack_1:output:09Decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*d
"Decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 Decoder/conv2d_transpose_1/stackPack1Decoder/conv2d_transpose_1/strided_slice:output:0+Decoder/conv2d_transpose_1/stack/1:output:0+Decoder/conv2d_transpose_1/stack/2:output:0+Decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*Decoder/conv2d_transpose_1/strided_slice_1StridedSlice)Decoder/conv2d_transpose_1/stack:output:09Decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+Decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_1/stack:output:0BDecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ј
1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
"Decoder/conv2d_transpose_1/BiasAddBiasAdd4Decoder/conv2d_transpose_1/conv2d_transpose:output:09Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Decoder/conv2d_transpose_1/EluElu+Decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*n
Decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   p
Decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_1/mulMul&Decoder/up_sampling2d_1/Const:output:0(Decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:№
4Decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_1/Elu:activations:0Decoder/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(o
Decoder/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      џ
(Decoder/resizing_1/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_1/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity9Decoder/resizing_1/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@o

Identity_1Identity/Encoder/dense_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ќ
NoOpNoOp2^Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp'^Decoder/dense_3/BiasAdd/ReadVariableOp&^Decoder/dense_3/MatMul/ReadVariableOp(^Encoder/conv2d_1/BiasAdd/ReadVariableOp'^Encoder/conv2d_1/Conv2D/ReadVariableOp'^Encoder/dense_2/BiasAdd/ReadVariableOp&^Encoder/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2f
1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2P
&Decoder/dense_3/BiasAdd/ReadVariableOp&Decoder/dense_3/BiasAdd/ReadVariableOp2N
%Decoder/dense_3/MatMul/ReadVariableOp%Decoder/dense_3/MatMul/ReadVariableOp2R
'Encoder/conv2d_1/BiasAdd/ReadVariableOp'Encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_1/Conv2D/ReadVariableOp&Encoder/conv2d_1/Conv2D/ReadVariableOp2P
&Encoder/dense_2/BiasAdd/ReadVariableOp&Encoder/dense_2/BiasAdd/ReadVariableOp2N
%Encoder/dense_2/MatMul/ReadVariableOp%Encoder/dense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1354814

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ

о
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354297
input_6!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354277x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	input_6
З
е
)__inference_Encoder_layer_call_fn_1354582

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_1353792o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*М
serving_defaultЈ
D
input_69
serving_default_input_6:0џџџџџџџџџ@D
Decoder9
StatefulPartitionedCall:0џџџџџџџџџ@tensorflow/serving/predict:єУ
у
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
З
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_sequential
Ф
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories"
_tf_keras_sequential
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object

3trace_0
4trace_1
5trace_2
6trace_32Ќ
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354251
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354297
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354392
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354414Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z3trace_0z4trace_1z5trace_2z6trace_3

7trace_0
8trace_1
9trace_2
:trace_32
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354180
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354204
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354491
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354568Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z7trace_0z8trace_1z9trace_2z:trace_3
ЭBЪ
"__inference__wrapped_model_1353657input_6"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ
;iter

<beta_1

=beta_2
	>decay
?learning_rate&mъ'mы(mь)mэ*mю+mя,m№-mё&vђ'vѓ(vє)vѕ*vі+vї,vј-vљ"
	optimizer
,
@serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper

A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

&kernel
'bias
#G_self_saveable_object_factories
 H_jit_compiled_convolution_op"
_tf_keras_layer
Ъ
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
#O_self_saveable_object_factories"
_tf_keras_layer
Ъ
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
#V_self_saveable_object_factories"
_tf_keras_layer
р
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

(kernel
)bias
#]_self_saveable_object_factories"
_tf_keras_layer
<
&0
'1
(2
)3"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
ctrace_0
dtrace_1
etrace_2
ftrace_32ф
)__inference_Encoder_layer_call_fn_1353804
)__inference_Encoder_layer_call_fn_1353843
)__inference_Encoder_layer_call_fn_1354582
)__inference_Encoder_layer_call_fn_1354596Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zctrace_0zdtrace_1zetrace_2zftrace_3
Л
gtrace_0
htrace_1
itrace_2
jtrace_32а
D__inference_Encoder_layer_call_and_return_conditional_losses_1353739
D__inference_Encoder_layer_call_and_return_conditional_losses_1353764
D__inference_Encoder_layer_call_and_return_conditional_losses_1354630
D__inference_Encoder_layer_call_and_return_conditional_losses_1354664Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zgtrace_0zhtrace_1zitrace_2zjtrace_3
 "
trackable_dict_wrapper
р
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

*kernel
+bias
#q_self_saveable_object_factories"
_tf_keras_layer
Ъ
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
#x_self_saveable_object_factories"
_tf_keras_layer

y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

,kernel
-bias
#_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories"
_tf_keras_layer
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories"
_tf_keras_layer
<
*0
+1
,2
-3"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
з
trace_0
trace_1
trace_2
trace_32ф
)__inference_Decoder_layer_call_fn_1354077
)__inference_Decoder_layer_call_fn_1354107
)__inference_Decoder_layer_call_fn_1354677
)__inference_Decoder_layer_call_fn_1354690Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
У
trace_0
trace_1
trace_2
trace_32а
D__inference_Decoder_layer_call_and_return_conditional_losses_1354029
D__inference_Decoder_layer_call_and_return_conditional_losses_1354046
D__inference_Decoder_layer_call_and_return_conditional_losses_1354737
D__inference_Decoder_layer_call_and_return_conditional_losses_1354784Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
 "
trackable_dict_wrapper
):'2conv2d_1/kernel
:2conv2d_1/bias
!:	  2dense_2/kernel
: 2dense_2/bias
!:	  2dense_3/kernel
: 2dense_3/bias
3:12conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354251input_6"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354297input_6"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354392inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354414inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354180input_6"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354204input_6"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354491inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354568inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЬBЩ
%__inference_signature_wrapper_1354370input_6"
В
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
annotationsЊ *
 
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ц
Ѓtrace_02Ч
*__inference_conv2d_1_layer_call_fn_1354793
В
FullArgSpec
args

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
annotationsЊ *
 zЃtrace_0

Єtrace_02т
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1354804
В
FullArgSpec
args

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
annotationsЊ *
 zЄtrace_0
 "
trackable_dict_wrapper
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
э
Њtrace_02Ю
1__inference_max_pooling2d_1_layer_call_fn_1354809
В
FullArgSpec
args

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
annotationsЊ *
 zЊtrace_0

Ћtrace_02щ
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1354814
В
FullArgSpec
args

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
annotationsЊ *
 zЋtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ч
Бtrace_02Ш
+__inference_flatten_1_layer_call_fn_1354819
В
FullArgSpec
args

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
annotationsЊ *
 zБtrace_0

Вtrace_02у
F__inference_flatten_1_layer_call_and_return_conditional_losses_1354825
В
FullArgSpec
args

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
annotationsЊ *
 zВtrace_0
 "
trackable_dict_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
Иactivity_regularizer_fn
*\&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
х
Кtrace_02Ц
)__inference_dense_2_layer_call_fn_1354834
В
FullArgSpec
args

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
annotationsЊ *
 zКtrace_0

Лtrace_02х
H__inference_dense_2_layer_call_and_return_all_conditional_losses_1354845
В
FullArgSpec
args

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
annotationsЊ *
 zЛtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
)__inference_Encoder_layer_call_fn_1353804input_4"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
)__inference_Encoder_layer_call_fn_1353843input_4"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
)__inference_Encoder_layer_call_fn_1354582inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
)__inference_Encoder_layer_call_fn_1354596inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_1353739input_4"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_1353764input_4"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_1354630inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Encoder_layer_call_and_return_conditional_losses_1354664inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
х
Сtrace_02Ц
)__inference_dense_3_layer_call_fn_1354865
В
FullArgSpec
args

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
annotationsЊ *
 zСtrace_0

Тtrace_02с
D__inference_dense_3_layer_call_and_return_conditional_losses_1354876
В
FullArgSpec
args

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
annotationsЊ *
 zТtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ч
Шtrace_02Ш
+__inference_reshape_1_layer_call_fn_1354881
В
FullArgSpec
args

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
annotationsЊ *
 zШtrace_0

Щtrace_02у
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354895
В
FullArgSpec
args

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
annotationsЊ *
 zЩtrace_0
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
№
Яtrace_02б
4__inference_conv2d_transpose_1_layer_call_fn_1354904
В
FullArgSpec
args

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
annotationsЊ *
 zЯtrace_0

аtrace_02ь
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1354942
В
FullArgSpec
args

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
annotationsЊ *
 zаtrace_0
 "
trackable_dict_wrapper
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
жtrace_02Ю
1__inference_up_sampling2d_1_layer_call_fn_1354947
В
FullArgSpec
args

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
annotationsЊ *
 zжtrace_0

зtrace_02щ
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1354959
В
FullArgSpec
args

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
annotationsЊ *
 zзtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш
нtrace_02Щ
,__inference_resizing_1_layer_call_fn_1354964
В
FullArgSpec
args

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
annotationsЊ *
 zнtrace_0

оtrace_02ф
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354970
В
FullArgSpec
args

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
annotationsЊ *
 zоtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
)__inference_Decoder_layer_call_fn_1354077input_5"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
)__inference_Decoder_layer_call_fn_1354107input_5"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
)__inference_Decoder_layer_call_fn_1354677inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
)__inference_Decoder_layer_call_fn_1354690inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_1354029input_5"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_1354046input_5"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_1354737inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_Decoder_layer_call_and_return_conditional_losses_1354784inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
п	variables
р	keras_api

сtotal

тcount"
_tf_keras_metric
c
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs"
_tf_keras_metric
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
дBб
*__inference_conv2d_1_layer_call_fn_1354793inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
яBь
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1354804inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
лBи
1__inference_max_pooling2d_1_layer_call_fn_1354809inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
іBѓ
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1354814inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
еBв
+__inference_flatten_1_layer_call_fn_1354819inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_flatten_1_layer_call_and_return_conditional_losses_1354825inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
ѓ
шtrace_02д
0__inference_dense_2_activity_regularizer_1353682
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	zшtrace_0

щtrace_02с
D__inference_dense_2_layer_call_and_return_conditional_losses_1354856
В
FullArgSpec
args

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
annotationsЊ *
 zщtrace_0
гBа
)__inference_dense_2_layer_call_fn_1354834inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ђBя
H__inference_dense_2_layer_call_and_return_all_conditional_losses_1354845inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
гBа
)__inference_dense_3_layer_call_fn_1354865inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_3_layer_call_and_return_conditional_losses_1354876inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
еBв
+__inference_reshape_1_layer_call_fn_1354881inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354895inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
оBл
4__inference_conv2d_transpose_1_layer_call_fn_1354904inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
љBі
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1354942inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
лBи
1__inference_up_sampling2d_1_layer_call_fn_1354947inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
іBѓ
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1354959inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
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
жBг
,__inference_resizing_1_layer_call_fn_1354964inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ёBю
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354970inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
0
с0
т1"
trackable_list_wrapper
.
п	variables"
_generic_user_object
:  (2total
:  (2count
0
х0
ц1"
trackable_list_wrapper
.
у	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
мBй
0__inference_dense_2_activity_regularizer_1353682x"
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
юBы
D__inference_dense_2_layer_call_and_return_conditional_losses_1354856inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
&:$	  2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
&:$	  2Adam/dense_3/kernel/m
 : 2Adam/dense_3/bias/m
8:62 Adam/conv2d_transpose_1/kernel/m
*:(2Adam/conv2d_transpose_1/bias/m
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
&:$	  2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
&:$	  2Adam/dense_3/kernel/v
 : 2Adam/dense_3/bias/v
8:62 Adam/conv2d_transpose_1/kernel/v
*:(2Adam/conv2d_transpose_1/bias/vє
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354180&'()*+,-AЂ>
7Ђ4
*'
input_6џџџџџџџџџ@
p

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 є
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354204&'()*+,-AЂ>
7Ђ4
*'
input_6џџџџџџџџџ@
p 

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 ѓ
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354491&'()*+,-@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 ѓ
V__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_1354568&'()*+,-@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 И
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354251y&'()*+,-AЂ>
7Ђ4
*'
input_6џџџџџџџџџ@
p

 
Њ "*'
unknownџџџџџџџџџ@И
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354297y&'()*+,-AЂ>
7Ђ4
*'
input_6џџџџџџџџџ@
p 

 
Њ "*'
unknownџџџџџџџџџ@З
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354392x&'()*+,-@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "*'
unknownџџџџџџџџџ@З
;__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_1354414x&'()*+,-@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "*'
unknownџџџџџџџџџ@П
D__inference_Decoder_layer_call_and_return_conditional_losses_1354029w*+,-8Ђ5
.Ђ+
!
input_5џџџџџџџџџ 
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 П
D__inference_Decoder_layer_call_and_return_conditional_losses_1354046w*+,-8Ђ5
.Ђ+
!
input_5џџџџџџџџџ 
p 

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 О
D__inference_Decoder_layer_call_and_return_conditional_losses_1354737v*+,-7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 О
D__inference_Decoder_layer_call_and_return_conditional_losses_1354784v*+,-7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p 

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 
)__inference_Decoder_layer_call_fn_1354077l*+,-8Ђ5
.Ђ+
!
input_5џџџџџџџџџ 
p

 
Њ "*'
unknownџџџџџџџџџ@
)__inference_Decoder_layer_call_fn_1354107l*+,-8Ђ5
.Ђ+
!
input_5џџџџџџџџџ 
p 

 
Њ "*'
unknownџџџџџџџџџ@
)__inference_Decoder_layer_call_fn_1354677k*+,-7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "*'
unknownџџџџџџџџџ@
)__inference_Decoder_layer_call_fn_1354690k*+,-7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p 

 
Њ "*'
unknownџџџџџџџџџ@е
D__inference_Encoder_layer_call_and_return_conditional_losses_1353739&'()AЂ>
7Ђ4
*'
input_4џџџџџџџџџ@
p

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 е
D__inference_Encoder_layer_call_and_return_conditional_losses_1353764&'()AЂ>
7Ђ4
*'
input_4џџџџџџџџџ@
p 

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 д
D__inference_Encoder_layer_call_and_return_conditional_losses_1354630&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 д
D__inference_Encoder_layer_call_and_return_conditional_losses_1354664&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 
)__inference_Encoder_layer_call_fn_1353804l&'()AЂ>
7Ђ4
*'
input_4џџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ 
)__inference_Encoder_layer_call_fn_1353843l&'()AЂ>
7Ђ4
*'
input_4џџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ 
)__inference_Encoder_layer_call_fn_1354582k&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ 
)__inference_Encoder_layer_call_fn_1354596k&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ Ј
"__inference__wrapped_model_1353657&'()*+,-9Ђ6
/Ђ,
*'
input_6џџџџџџџџџ@
Њ ":Њ7
5
Decoder*'
decoderџџџџџџџџџ@Н
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1354804t&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ*
 
*__inference_conv2d_1_layer_call_fn_1354793i&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ ")&
unknownџџџџџџџџџ*ы
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1354942,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
4__inference_conv2d_transpose_1_layer_call_fn_1354904,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџc
0__inference_dense_2_activity_regularizer_1353682/Ђ
Ђ
	
x
Њ "
unknown Х
H__inference_dense_2_layer_call_and_return_all_conditional_losses_1354845y()0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 Ќ
D__inference_dense_2_layer_call_and_return_conditional_losses_1354856d()0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
)__inference_dense_2_layer_call_fn_1354834Y()0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџ Ќ
D__inference_dense_3_layer_call_and_return_conditional_losses_1354876d*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
)__inference_dense_3_layer_call_fn_1354865Y*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџ В
F__inference_flatten_1_layer_call_and_return_conditional_losses_1354825h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
+__inference_flatten_1_layer_call_fn_1354819]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ і
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1354814ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_max_pooling2d_1_layer_call_fn_1354809RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџВ
F__inference_reshape_1_layer_call_and_return_conditional_losses_1354895h0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
+__inference_reshape_1_layer_call_fn_1354881]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџЮ
G__inference_resizing_1_layer_call_and_return_conditional_losses_1354970IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 Ї
,__inference_resizing_1_layer_call_fn_1354964wIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџ@Ж
%__inference_signature_wrapper_1354370&'()*+,-DЂA
Ђ 
:Њ7
5
input_6*'
input_6џџџџџџџџџ@":Њ7
5
Decoder*'
decoderџџџџџџџџџ@і
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1354959ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_up_sampling2d_1_layer_call_fn_1354947RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ