Им
ьЯ
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
 "serve*2.12.02unknown8лџ

Adam/conv2d_transpose_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_9/bias/v

2Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/v*
_output_shapes
:*
dtype0
Є
 Adam/conv2d_transpose_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_9/kernel/v

4Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/v*&
_output_shapes
:*
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/v
z
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes	
: *
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	  *
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
: *
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes
:	  *
dtype0

Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_9/kernel/v

*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_transpose_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_9/bias/m

2Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/m*
_output_shapes
:*
dtype0
Є
 Adam/conv2d_transpose_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_9/kernel/m

4Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/m*&
_output_shapes
:*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/m
z
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes	
: *
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	  *
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
: *
dtype0

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes
:	  *
dtype0

Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_9/kernel/m

*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
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
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_9/bias

+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes
:*
dtype0

conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_9/kernel

-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*&
_output_shapes
:*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
: *
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	  *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
: *
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	  *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0

serving_default_input_21Placeholder*0
_output_shapes
:џџџџџџџџџ@*
dtype0*%
shape:џџџџџџџџџ@
р
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_21conv2d_9/kernelconv2d_9/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasconv2d_transpose_9/kernelconv2d_transpose_9/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_320084

NoOpNoOp
пX
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*X
valueXBX BX
Ї
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
signatures*
* 
ј
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
<
"0
#1
$2
%3
&4
'5
(6
)7*
<
"0
#1
$2
%3
&4
'5
(6
)7*
* 
А
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
* 
ф
7iter

8beta_1

9beta_2
	:decay
;learning_rate"mн#mо$mп%mр&mс'mт(mу)mф"vх#vц$vч%vш&vщ'vъ(vы)vь*

<serving_default* 
Ш
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

"kernel
#bias
 C_jit_compiled_convolution_op*

D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
І
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

$kernel
%bias*
 
"0
#1
$2
%3*
 
"0
#1
$2
%3*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
[trace_0
\trace_1
]trace_2
^trace_3* 
6
_trace_0
`trace_1
atrace_2
btrace_3* 
І
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

&kernel
'bias*

i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
Ш
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

(kernel
)bias
 u_jit_compiled_convolution_op*

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
OI
VARIABLE_VALUEconv2d_9/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_9/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_12/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_12/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_13/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_13/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_9/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose_9/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

0
1*
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
"0
#1*

"0
#1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

Єtrace_0* 

Ѕtrace_0* 

$0
%1*

$0
%1*
* 
З
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
Ћactivity_regularizer_fn
*U&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses*

­trace_0* 

Ўtrace_0* 
* 
 
0
1
2
3*
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
&0
'1*

&0
'1*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
* 
* 
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 

(0
)1*

(0
)1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

Тtrace_0* 

Уtrace_0* 
* 
* 
* 
* 

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

Щtrace_0* 

Ъtrace_0* 
* 
* 
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

аtrace_0* 

бtrace_0* 
* 
'
0
1
2
3
4*
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
в	variables
г	keras_api

дtotal

еcount*
M
ж	variables
з	keras_api

иtotal

йcount
к
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
лtrace_0* 

мtrace_0* 
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
д0
е1*

в	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

и0
й1*

ж	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_9/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_9/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_12/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_12/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_13/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_13/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_9/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_9/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_12/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_12/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_13/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_13/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Э
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasconv2d_transpose_9/kernelconv2d_transpose_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m Adam/conv2d_transpose_9/kernel/mAdam/conv2d_transpose_9/bias/mAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v Adam/conv2d_transpose_9/kernel/vAdam/conv2d_transpose_9/bias/vConst*.
Tin'
%2#*
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
__inference__traced_save_320905
Ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasconv2d_transpose_9/kernelconv2d_transpose_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m Adam/conv2d_transpose_9/kernel/mAdam/conv2d_transpose_9/bias/mAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v Adam/conv2d_transpose_9/kernel/vAdam/conv2d_transpose_9/bias/v*-
Tin&
$2"*
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
"__inference__traced_restore_321014аб
хg
Ћ
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320282

inputsI
/encoder_conv2d_9_conv2d_readvariableop_resource:>
0encoder_conv2d_9_biasadd_readvariableop_resource:B
/encoder_dense_12_matmul_readvariableop_resource:	  >
0encoder_dense_12_biasadd_readvariableop_resource: B
/decoder_dense_13_matmul_readvariableop_resource:	  ?
0decoder_dense_13_biasadd_readvariableop_resource:	 ]
Cdecoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_9_biasadd_readvariableop_resource:
identity

identity_1Ђ1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpЂ:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpЂ'Decoder/dense_13/BiasAdd/ReadVariableOpЂ&Decoder/dense_13/MatMul/ReadVariableOpЂ'Encoder/conv2d_9/BiasAdd/ReadVariableOpЂ&Encoder/conv2d_9/Conv2D/ReadVariableOpЂ'Encoder/dense_12/BiasAdd/ReadVariableOpЂ&Encoder/dense_12/MatMul/ReadVariableOp
&Encoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0М
Encoder/conv2d_9/Conv2DConv2Dinputs.Encoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

'Encoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Encoder/conv2d_9/BiasAddBiasAdd Encoder/conv2d_9/Conv2D:output:0/Encoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*x
Encoder/conv2d_9/EluElu!Encoder/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*К
Encoder/max_pooling2d_9/MaxPoolMaxPool"Encoder/conv2d_9/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ѓ
Encoder/flatten_6/ReshapeReshape(Encoder/max_pooling2d_9/MaxPool:output:0 Encoder/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&Encoder/dense_12/MatMul/ReadVariableOpReadVariableOp/encoder_dense_12_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ї
Encoder/dense_12/MatMulMatMul"Encoder/flatten_6/Reshape:output:0.Encoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'Encoder/dense_12/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
Encoder/dense_12/BiasAddBiasAdd!Encoder/dense_12/MatMul:product:0/Encoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Encoder/dense_12/EluElu!Encoder/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(Encoder/dense_12/ActivityRegularizer/AbsAbs"Encoder/dense_12/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ {
*Encoder/dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Г
(Encoder/dense_12/ActivityRegularizer/SumSum,Encoder/dense_12/ActivityRegularizer/Abs:y:03Encoder/dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*Encoder/dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8И
(Encoder/dense_12/ActivityRegularizer/mulMul3Encoder/dense_12/ActivityRegularizer/mul/x:output:01Encoder/dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
*Encoder/dense_12/ActivityRegularizer/ShapeShape"Encoder/dense_12/Elu:activations:0*
T0*
_output_shapes
::эЯ
8Encoder/dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:Encoder/dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:Encoder/dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2Encoder/dense_12/ActivityRegularizer/strided_sliceStridedSlice3Encoder/dense_12/ActivityRegularizer/Shape:output:0AEncoder/dense_12/ActivityRegularizer/strided_slice/stack:output:0CEncoder/dense_12/ActivityRegularizer/strided_slice/stack_1:output:0CEncoder/dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)Encoder/dense_12/ActivityRegularizer/CastCast;Encoder/dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Е
,Encoder/dense_12/ActivityRegularizer/truedivRealDiv,Encoder/dense_12/ActivityRegularizer/mul:z:0-Encoder/dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
&Decoder/dense_13/MatMul/ReadVariableOpReadVariableOp/decoder_dense_13_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ј
Decoder/dense_13/MatMulMatMul"Encoder/dense_12/Elu:activations:0.Decoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
'Decoder/dense_13/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_13_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Њ
Decoder/dense_13/BiasAddBiasAdd!Decoder/dense_13/MatMul:product:0/Decoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ q
Decoder/dense_13/EluElu!Decoder/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ w
Decoder/reshape_6/ShapeShape"Decoder/dense_13/Elu:activations:0*
T0*
_output_shapes
::эЯo
%Decoder/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
Decoder/reshape_6/strided_sliceStridedSlice Decoder/reshape_6/Shape:output:0.Decoder/reshape_6/strided_slice/stack:output:00Decoder/reshape_6/strided_slice/stack_1:output:00Decoder/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Decoder/reshape_6/Reshape/shapePack(Decoder/reshape_6/strided_slice:output:0*Decoder/reshape_6/Reshape/shape/1:output:0*Decoder/reshape_6/Reshape/shape/2:output:0*Decoder/reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ќ
Decoder/reshape_6/ReshapeReshape"Decoder/dense_13/Elu:activations:0(Decoder/reshape_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 Decoder/conv2d_transpose_9/ShapeShape"Decoder/reshape_6/Reshape:output:0*
T0*
_output_shapes
::эЯx
.Decoder/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(Decoder/conv2d_transpose_9/strided_sliceStridedSlice)Decoder/conv2d_transpose_9/Shape:output:07Decoder/conv2d_transpose_9/strided_slice/stack:output:09Decoder/conv2d_transpose_9/strided_slice/stack_1:output:09Decoder/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*d
"Decoder/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 Decoder/conv2d_transpose_9/stackPack1Decoder/conv2d_transpose_9/strided_slice:output:0+Decoder/conv2d_transpose_9/stack/1:output:0+Decoder/conv2d_transpose_9/stack/2:output:0+Decoder/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*Decoder/conv2d_transpose_9/strided_slice_1StridedSlice)Decoder/conv2d_transpose_9/stack:output:09Decoder/conv2d_transpose_9/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_9/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+Decoder/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_9/stack:output:0BDecoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_6/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ј
1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
"Decoder/conv2d_transpose_9/BiasAddBiasAdd4Decoder/conv2d_transpose_9/conv2d_transpose:output:09Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Decoder/conv2d_transpose_9/EluElu+Decoder/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*n
Decoder/up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   p
Decoder/up_sampling2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_9/mulMul&Decoder/up_sampling2d_9/Const:output:0(Decoder/up_sampling2d_9/Const_1:output:0*
T0*
_output_shapes
:№
4Decoder/up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_9/Elu:activations:0Decoder/up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(o
Decoder/resizing_6/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      џ
(Decoder/resizing_6/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_6/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity9Decoder/resizing_6/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@p

Identity_1Identity0Encoder/dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: А
NoOpNoOp2^Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp(^Decoder/dense_13/BiasAdd/ReadVariableOp'^Decoder/dense_13/MatMul/ReadVariableOp(^Encoder/conv2d_9/BiasAdd/ReadVariableOp'^Encoder/conv2d_9/Conv2D/ReadVariableOp(^Encoder/dense_12/BiasAdd/ReadVariableOp'^Encoder/dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2f
1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2R
'Decoder/dense_13/BiasAdd/ReadVariableOp'Decoder/dense_13/BiasAdd/ReadVariableOp2P
&Decoder/dense_13/MatMul/ReadVariableOp&Decoder/dense_13/MatMul/ReadVariableOp2R
'Encoder/conv2d_9/BiasAdd/ReadVariableOp'Encoder/conv2d_9/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_9/Conv2D/ReadVariableOp&Encoder/conv2d_9/Conv2D/ReadVariableOp2R
'Encoder/dense_12/BiasAdd/ReadVariableOp'Encoder/dense_12/BiasAdd/ReadVariableOp2P
&Encoder/dense_12/MatMul/ReadVariableOp&Encoder/dense_12/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
л

!__inference__wrapped_model_319371
input_21c
Iae_conv_prep_flatten_stft_encoder_conv2d_9_conv2d_readvariableop_resource:X
Jae_conv_prep_flatten_stft_encoder_conv2d_9_biasadd_readvariableop_resource:\
Iae_conv_prep_flatten_stft_encoder_dense_12_matmul_readvariableop_resource:	  X
Jae_conv_prep_flatten_stft_encoder_dense_12_biasadd_readvariableop_resource: \
Iae_conv_prep_flatten_stft_decoder_dense_13_matmul_readvariableop_resource:	  Y
Jae_conv_prep_flatten_stft_decoder_dense_13_biasadd_readvariableop_resource:	 w
]ae_conv_prep_flatten_stft_decoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:b
Tae_conv_prep_flatten_stft_decoder_conv2d_transpose_9_biasadd_readvariableop_resource:
identityЂKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpЂTAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpЂAAE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOpЂAAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOpЂAAE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOpЂ@AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOpв
@AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_encoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ђ
1AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2DConv2Dinput_21HAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ш
AAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_stft_encoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
2AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAddBiasAdd:AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D:output:0IAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*Ќ
.AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/EluElu;AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*ю
9AE_Conv_prep_flatten_STFT/Encoder/max_pooling2d_9/MaxPoolMaxPool<AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

1AE_Conv_prep_flatten_STFT/Encoder/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ё
3AE_Conv_prep_flatten_STFT/Encoder/flatten_6/ReshapeReshapeBAE_Conv_prep_flatten_STFT/Encoder/max_pooling2d_9/MaxPool:output:0:AE_Conv_prep_flatten_STFT/Encoder/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Ы
@AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_encoder_dense_12_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0ѕ
1AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMulMatMul<AE_Conv_prep_flatten_STFT/Encoder/flatten_6/Reshape:output:0HAE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ш
AAE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_stft_encoder_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ї
2AE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAddBiasAdd;AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul:product:0IAE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
.AE_Conv_prep_flatten_STFT/Encoder/dense_12/EluElu;AE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Й
BAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/AbsAbs<AE_Conv_prep_flatten_STFT/Encoder/dense_12/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 
DAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
BAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/SumSumFAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/Abs:y:0MAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 
DAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
BAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/mulMulMAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/mul/x:output:0KAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: О
DAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/ShapeShape<AE_Conv_prep_flatten_STFT/Encoder/dense_12/Elu:activations:0*
T0*
_output_shapes
::эЯ
RAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
TAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
TAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
LAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_sliceStridedSliceMAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/Shape:output:0[AE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stack:output:0]AE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stack_1:output:0]AE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskв
CAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/CastCastUAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
FAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/truedivRealDivFAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/mul:z:0GAE_Conv_prep_flatten_STFT/Encoder/dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ы
@AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOpReadVariableOpIae_conv_prep_flatten_stft_decoder_dense_13_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0і
1AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMulMatMul<AE_Conv_prep_flatten_STFT/Encoder/dense_12/Elu:activations:0HAE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ Щ
AAE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOpReadVariableOpJae_conv_prep_flatten_stft_decoder_dense_13_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0ј
2AE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAddBiasAdd;AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul:product:0IAE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ Ѕ
.AE_Conv_prep_flatten_STFT/Decoder/dense_13/EluElu;AE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Ћ
1AE_Conv_prep_flatten_STFT/Decoder/reshape_6/ShapeShape<AE_Conv_prep_flatten_STFT/Decoder/dense_13/Elu:activations:0*
T0*
_output_shapes
::эЯ
?AE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9AE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_sliceStridedSlice:AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Shape:output:0HAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stack:output:0JAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stack_1:output:0JAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}
;AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
9AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shapePackBAE_Conv_prep_flatten_STFT/Decoder/reshape_6/strided_slice:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/1:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/2:output:0DAE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:њ
3AE_Conv_prep_flatten_STFT/Decoder/reshape_6/ReshapeReshape<AE_Conv_prep_flatten_STFT/Decoder/dense_13/Elu:activations:0BAE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџД
:AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/ShapeShape<AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape:output:0*
T0*
_output_shapes
::эЯ
HAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
BAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_sliceStridedSliceCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/Shape:output:0QAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stack:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stack_1:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*~
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
:AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stackPackKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/1:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/2:output:0EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:
JAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
LAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
LAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
DAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1StridedSliceCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stack:output:0UAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stack_1:output:0UAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
TAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp]ae_conv_prep_flatten_stft_decoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0 
EAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transposeConv2DBackpropInputCAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/stack:output:0\AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0<AE_Conv_prep_flatten_STFT/Decoder/reshape_6/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
м
KAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOpTae_conv_prep_flatten_stft_decoder_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
<AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAddBiasAddNAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose:output:0SAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*Р
8AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/EluEluEAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
7AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   
9AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ч
5AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/mulMul@AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/Const:output:0BAE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/Const_1:output:0*
T0*
_output_shapes
:О
NAE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighborFAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/Elu:activations:09AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(
8AE_Conv_prep_flatten_STFT/Decoder/resizing_6/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      Э
BAE_Conv_prep_flatten_STFT/Decoder/resizing_6/resize/ResizeBilinearResizeBilinear_AE_Conv_prep_flatten_STFT/Decoder/up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0AAE_Conv_prep_flatten_STFT/Decoder/resizing_6/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(Ћ
IdentityIdentitySAE_Conv_prep_flatten_STFT/Decoder/resizing_6/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOpL^AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpU^AE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpB^AE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOpB^AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOpB^AE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOpA^AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2
KAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpKAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp2Ќ
TAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpTAE_Conv_prep_flatten_STFT/Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2
AAE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_STFT/Decoder/dense_13/BiasAdd/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOp@AE_Conv_prep_flatten_STFT/Decoder/dense_13/MatMul/ReadVariableOp2
AAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_STFT/Encoder/conv2d_9/BiasAdd/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOp@AE_Conv_prep_flatten_STFT/Encoder/conv2d_9/Conv2D/ReadVariableOp2
AAE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOpAAE_Conv_prep_flatten_STFT/Encoder/dense_12/BiasAdd/ReadVariableOp2
@AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOp@AE_Conv_prep_flatten_STFT/Encoder/dense_12/MatMul/ReadVariableOp:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21

§
D__inference_conv2d_9_layer_call_and_return_conditional_losses_320518

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
м
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726

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
Б
F
*__inference_flatten_6_layer_call_fn_320533

inputs
identityБ
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
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424a
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

§
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411

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
 

о
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_319965
input_21!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319945x
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
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21
 

о
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320011
input_21!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:џџџџџџџџџ@: **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319991x
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
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21
ђ
ж
C__inference_Decoder_layer_call_and_return_conditional_losses_319760
input_20"
dense_13_319746:	  
dense_13_319748:	 3
conv2d_transpose_9_319752:'
conv2d_transpose_9_319754:
identityЂ*conv2d_transpose_9/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallѓ
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_13_319746dense_13_319748*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_319706х
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726М
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_transpose_9_319752conv2d_transpose_9_319754*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662
up_sampling2d_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685ч
resizing_6/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740{
IdentityIdentity#resizing_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_9/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
input_20
Ж
з
(__inference_Decoder_layer_call_fn_319821
input_20
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319810x
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
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
input_20
Х

)__inference_dense_12_layer_call_fn_320548

inputs
unknown:	  
	unknown_0: 
identityЂStatefulPartitionedCallй
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437o
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
м
a
E__inference_reshape_6_layer_call_and_return_conditional_losses_320609

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
И
ж
(__inference_Encoder_layer_call_fn_319557
input_19!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319545o
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
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_19
А
е
(__inference_Decoder_layer_call_fn_320404

inputs
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallћ
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
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319810x
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


і
D__inference_dense_12_layer_call_and_return_conditional_losses_319437

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
Ђ
П
"__inference__traced_restore_321014
file_prefix:
 assignvariableop_conv2d_9_kernel:.
 assignvariableop_1_conv2d_9_bias:5
"assignvariableop_2_dense_12_kernel:	  .
 assignvariableop_3_dense_12_bias: 5
"assignvariableop_4_dense_13_kernel:	  /
 assignvariableop_5_dense_13_bias:	 F
,assignvariableop_6_conv2d_transpose_9_kernel:8
*assignvariableop_7_conv2d_transpose_9_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: D
*assignvariableop_17_adam_conv2d_9_kernel_m:6
(assignvariableop_18_adam_conv2d_9_bias_m:=
*assignvariableop_19_adam_dense_12_kernel_m:	  6
(assignvariableop_20_adam_dense_12_bias_m: =
*assignvariableop_21_adam_dense_13_kernel_m:	  7
(assignvariableop_22_adam_dense_13_bias_m:	 N
4assignvariableop_23_adam_conv2d_transpose_9_kernel_m:@
2assignvariableop_24_adam_conv2d_transpose_9_bias_m:D
*assignvariableop_25_adam_conv2d_9_kernel_v:6
(assignvariableop_26_adam_conv2d_9_bias_v:=
*assignvariableop_27_adam_dense_12_kernel_v:	  6
(assignvariableop_28_adam_dense_12_bias_v: =
*assignvariableop_29_adam_dense_13_kernel_v:	  7
(assignvariableop_30_adam_dense_13_bias_v:	 N
4assignvariableop_31_adam_conv2d_transpose_9_kernel_v:@
2assignvariableop_32_adam_conv2d_transpose_9_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_13_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_13_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_9_biasIdentity_7:output:0"/device:CPU:0*&
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_9_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_9_bias_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_12_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_12_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_13_kernel_mIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_13_bias_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_conv2d_transpose_9_kernel_mIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_conv2d_transpose_9_bias_mIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_9_kernel_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_9_bias_vIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_12_kernel_vIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_12_bias_vIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_13_kernel_vIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_13_bias_vIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_conv2d_transpose_9_kernel_vIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_conv2d_transpose_9_bias_vIdentity_32:output:0"/device:CPU:0*&
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


і
D__inference_dense_12_layer_call_and_return_conditional_losses_320570

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
 

ї
D__inference_dense_13_layer_call_and_return_conditional_losses_320590

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
хg
Ћ
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320205

inputsI
/encoder_conv2d_9_conv2d_readvariableop_resource:>
0encoder_conv2d_9_biasadd_readvariableop_resource:B
/encoder_dense_12_matmul_readvariableop_resource:	  >
0encoder_dense_12_biasadd_readvariableop_resource: B
/decoder_dense_13_matmul_readvariableop_resource:	  ?
0decoder_dense_13_biasadd_readvariableop_resource:	 ]
Cdecoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_9_biasadd_readvariableop_resource:
identity

identity_1Ђ1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpЂ:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpЂ'Decoder/dense_13/BiasAdd/ReadVariableOpЂ&Decoder/dense_13/MatMul/ReadVariableOpЂ'Encoder/conv2d_9/BiasAdd/ReadVariableOpЂ&Encoder/conv2d_9/Conv2D/ReadVariableOpЂ'Encoder/dense_12/BiasAdd/ReadVariableOpЂ&Encoder/dense_12/MatMul/ReadVariableOp
&Encoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0М
Encoder/conv2d_9/Conv2DConv2Dinputs.Encoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

'Encoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Encoder/conv2d_9/BiasAddBiasAdd Encoder/conv2d_9/Conv2D:output:0/Encoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*x
Encoder/conv2d_9/EluElu!Encoder/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*К
Encoder/max_pooling2d_9/MaxPoolMaxPool"Encoder/conv2d_9/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
h
Encoder/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ѓ
Encoder/flatten_6/ReshapeReshape(Encoder/max_pooling2d_9/MaxPool:output:0 Encoder/flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&Encoder/dense_12/MatMul/ReadVariableOpReadVariableOp/encoder_dense_12_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ї
Encoder/dense_12/MatMulMatMul"Encoder/flatten_6/Reshape:output:0.Encoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'Encoder/dense_12/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
Encoder/dense_12/BiasAddBiasAdd!Encoder/dense_12/MatMul:product:0/Encoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Encoder/dense_12/EluElu!Encoder/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(Encoder/dense_12/ActivityRegularizer/AbsAbs"Encoder/dense_12/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ {
*Encoder/dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Г
(Encoder/dense_12/ActivityRegularizer/SumSum,Encoder/dense_12/ActivityRegularizer/Abs:y:03Encoder/dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*Encoder/dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8И
(Encoder/dense_12/ActivityRegularizer/mulMul3Encoder/dense_12/ActivityRegularizer/mul/x:output:01Encoder/dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
*Encoder/dense_12/ActivityRegularizer/ShapeShape"Encoder/dense_12/Elu:activations:0*
T0*
_output_shapes
::эЯ
8Encoder/dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:Encoder/dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:Encoder/dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2Encoder/dense_12/ActivityRegularizer/strided_sliceStridedSlice3Encoder/dense_12/ActivityRegularizer/Shape:output:0AEncoder/dense_12/ActivityRegularizer/strided_slice/stack:output:0CEncoder/dense_12/ActivityRegularizer/strided_slice/stack_1:output:0CEncoder/dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)Encoder/dense_12/ActivityRegularizer/CastCast;Encoder/dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Е
,Encoder/dense_12/ActivityRegularizer/truedivRealDiv,Encoder/dense_12/ActivityRegularizer/mul:z:0-Encoder/dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
&Decoder/dense_13/MatMul/ReadVariableOpReadVariableOp/decoder_dense_13_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0Ј
Decoder/dense_13/MatMulMatMul"Encoder/dense_12/Elu:activations:0.Decoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
'Decoder/dense_13/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_13_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0Њ
Decoder/dense_13/BiasAddBiasAdd!Decoder/dense_13/MatMul:product:0/Decoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ q
Decoder/dense_13/EluElu!Decoder/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ w
Decoder/reshape_6/ShapeShape"Decoder/dense_13/Elu:activations:0*
T0*
_output_shapes
::эЯo
%Decoder/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Decoder/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Decoder/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
Decoder/reshape_6/strided_sliceStridedSlice Decoder/reshape_6/Shape:output:0.Decoder/reshape_6/strided_slice/stack:output:00Decoder/reshape_6/strided_slice/stack_1:output:00Decoder/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!Decoder/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!Decoder/reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Decoder/reshape_6/Reshape/shapePack(Decoder/reshape_6/strided_slice:output:0*Decoder/reshape_6/Reshape/shape/1:output:0*Decoder/reshape_6/Reshape/shape/2:output:0*Decoder/reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ќ
Decoder/reshape_6/ReshapeReshape"Decoder/dense_13/Elu:activations:0(Decoder/reshape_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 Decoder/conv2d_transpose_9/ShapeShape"Decoder/reshape_6/Reshape:output:0*
T0*
_output_shapes
::эЯx
.Decoder/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0Decoder/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0Decoder/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(Decoder/conv2d_transpose_9/strided_sliceStridedSlice)Decoder/conv2d_transpose_9/Shape:output:07Decoder/conv2d_transpose_9/strided_slice/stack:output:09Decoder/conv2d_transpose_9/strided_slice/stack_1:output:09Decoder/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"Decoder/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"Decoder/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*d
"Decoder/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 Decoder/conv2d_transpose_9/stackPack1Decoder/conv2d_transpose_9/strided_slice:output:0+Decoder/conv2d_transpose_9/stack/1:output:0+Decoder/conv2d_transpose_9/stack/2:output:0+Decoder/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:z
0Decoder/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Decoder/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Decoder/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*Decoder/conv2d_transpose_9/strided_slice_1StridedSlice)Decoder/conv2d_transpose_9/stack:output:09Decoder/conv2d_transpose_9/strided_slice_1/stack:output:0;Decoder/conv2d_transpose_9/strided_slice_1/stack_1:output:0;Decoder/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+Decoder/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)Decoder/conv2d_transpose_9/stack:output:0BDecoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0"Decoder/reshape_6/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides
Ј
1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0и
"Decoder/conv2d_transpose_9/BiasAddBiasAdd4Decoder/conv2d_transpose_9/conv2d_transpose:output:09Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
Decoder/conv2d_transpose_9/EluElu+Decoder/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*n
Decoder/up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   p
Decoder/up_sampling2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
Decoder/up_sampling2d_9/mulMul&Decoder/up_sampling2d_9/Const:output:0(Decoder/up_sampling2d_9/Const_1:output:0*
T0*
_output_shapes
:№
4Decoder/up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor,Decoder/conv2d_transpose_9/Elu:activations:0Decoder/up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(o
Decoder/resizing_6/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      џ
(Decoder/resizing_6/resize/ResizeBilinearResizeBilinearEDecoder/up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0'Decoder/resizing_6/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity9Decoder/resizing_6/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@p

Identity_1Identity0Encoder/dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: А
NoOpNoOp2^Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp;^Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp(^Decoder/dense_13/BiasAdd/ReadVariableOp'^Decoder/dense_13/MatMul/ReadVariableOp(^Encoder/conv2d_9/BiasAdd/ReadVariableOp'^Encoder/conv2d_9/Conv2D/ReadVariableOp(^Encoder/dense_12/BiasAdd/ReadVariableOp'^Encoder/dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ@: : : : : : : : 2f
1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp1Decoder/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Decoder/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2R
'Decoder/dense_13/BiasAdd/ReadVariableOp'Decoder/dense_13/BiasAdd/ReadVariableOp2P
&Decoder/dense_13/MatMul/ReadVariableOp&Decoder/dense_13/MatMul/ReadVariableOp2R
'Encoder/conv2d_9/BiasAdd/ReadVariableOp'Encoder/conv2d_9/BiasAdd/ReadVariableOp2P
&Encoder/conv2d_9/Conv2D/ReadVariableOp&Encoder/conv2d_9/Conv2D/ReadVariableOp2R
'Encoder/dense_12/BiasAdd/ReadVariableOp'Encoder/dense_12/BiasAdd/ReadVariableOp2P
&Encoder/dense_12/MatMul/ReadVariableOp&Encoder/dense_12/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ч
Ј
3__inference_conv2d_transpose_9_layer_call_fn_320618

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall§
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
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662
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
Б
F
*__inference_reshape_6_layer_call_fn_320595

inputs
identityИ
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
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726h
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
Ч
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_320539

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
Л
л
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319945

inputs(
encoder_319924:
encoder_319926:!
encoder_319928:	  
encoder_319930: !
decoder_319934:	  
decoder_319936:	 (
decoder_319938:
decoder_319940:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_319924encoder_319926encoder_319928encoder_319930*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319506Л
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_319934decoder_319936decoder_319938decoder_319940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319780
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
ь

)__inference_conv2d_9_layer_call_fn_320507

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411w
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
В
д
(__inference_Encoder_layer_call_fn_320310

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallѕ
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
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319545o
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
С
н
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319918
input_21(
encoder_319897:
encoder_319899:!
encoder_319901:	  
encoder_319903: !
decoder_319907:	  
decoder_319909:	 (
decoder_319911:
decoder_319913:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_21encoder_319897encoder_319899encoder_319901encoder_319903*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319545Л
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_319907decoder_319909decoder_319911decoder_319913*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319810
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
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21
И
L
0__inference_up_sampling2d_9_layer_call_fn_320661

inputs
identityй
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
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685
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
г	
Ш
$__inference_signature_wrapper_320084
input_21!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_319371x
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
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21
Ю
b
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740

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
Л
л
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319991

inputs(
encoder_319970:
encoder_319972:!
encoder_319974:	  
encoder_319976: !
decoder_319980:	  
decoder_319982:	 (
decoder_319984:
decoder_319986:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_319970encoder_319972encoder_319974encoder_319976*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319545Л
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_319980decoder_319982decoder_319984decoder_319986*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319810
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


м
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320106

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallФ
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

*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319945x
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
њ#

N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_320656

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
Ж
з
(__inference_Decoder_layer_call_fn_319791
input_20
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319780x
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
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
input_20


м
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320128

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
	unknown_3:	  
	unknown_4:	 #
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallФ
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

*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319991x
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
Л'
ч
C__inference_Encoder_layer_call_and_return_conditional_losses_320344

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource::
'dense_12_matmul_readvariableop_resource:	  6
(dense_12_biasadd_readvariableop_resource: 
identity

identity_1Ђconv2d_9/BiasAdd/ReadVariableOpЂconv2d_9/Conv2D/ReadVariableOpЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ќ
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
conv2d_9/EluEluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Њ
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_6/ReshapeReshape max_pooling2d_9/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_12/EluEludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ u
 dense_12/ActivityRegularizer/AbsAbsdense_12/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ s
"dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_12/ActivityRegularizer/SumSum$dense_12/ActivityRegularizer/Abs:y:0+dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8 
 dense_12/ActivityRegularizer/mulMul+dense_12/ActivityRegularizer/mul/x:output:0)dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: z
"dense_12/ActivityRegularizer/ShapeShapedense_12/Elu:activations:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_12/ActivityRegularizer/truedivRealDiv$dense_12/ActivityRegularizer/mul:z:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: i
IdentityIdentitydense_12/Elu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ь
NoOpNoOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ч
G
+__inference_resizing_6_layer_call_fn_320678

inputs
identityК
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
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740i
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
 

ї
D__inference_dense_13_layer_call_and_return_conditional_losses_319706

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
А
е
(__inference_Decoder_layer_call_fn_320391

inputs
unknown:	  
	unknown_0:	 #
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallћ
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
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319780x
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
Ч
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424

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
З8

C__inference_Decoder_layer_call_and_return_conditional_losses_320498

inputs:
'dense_13_matmul_readvariableop_resource:	  7
(dense_13_biasadd_readvariableop_resource:	 U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identityЂ)conv2d_transpose_9/BiasAdd/ReadVariableOpЂ2conv2d_transpose_9/conv2d_transpose/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOp
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0|
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ a
dense_13/EluEludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ g
reshape_6/ShapeShapedense_13/Elu:activations:0*
T0*
_output_shapes
::эЯg
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0"reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshapedense_13/Elu:activations:0 reshape_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
conv2d_transpose_9/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
::эЯp
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*\
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0reshape_6/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*|
conv2d_transpose_9/EluElu#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*f
up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   h
up_sampling2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_9/mulMulup_sampling2d_9/Const:output:0 up_sampling2d_9/Const_1:output:0*
T0*
_output_shapes
:и
,up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_9/Elu:activations:0up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(g
resizing_6/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      ч
 resizing_6/resize/ResizeBilinearResizeBilinear=up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0resizing_6/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity1resizing_6/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ъ
NoOpNoOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
З8

C__inference_Decoder_layer_call_and_return_conditional_losses_320451

inputs:
'dense_13_matmul_readvariableop_resource:	  7
(dense_13_biasadd_readvariableop_resource:	 U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identityЂ)conv2d_transpose_9/BiasAdd/ReadVariableOpЂ2conv2d_transpose_9/conv2d_transpose/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOp
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0|
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ a
dense_13/EluEludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ g
reshape_6/ShapeShapedense_13/Elu:activations:0*
T0*
_output_shapes
::эЯg
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0"reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshapedense_13/Elu:activations:0 reshape_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
conv2d_transpose_9/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
::эЯp
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*\
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0reshape_6/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*|
conv2d_transpose_9/EluElu#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*f
up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"   *   h
up_sampling2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_9/mulMulup_sampling2d_9/Const:output:0 up_sampling2d_9/Const_1:output:0*
T0*
_output_shapes
:и
,up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor$conv2d_transpose_9/Elu:activations:0up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ?~*
half_pixel_centers(g
resizing_6/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@      ч
 resizing_6/resize/ResizeBilinearResizeBilinear=up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0resizing_6/resize/size:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(
IdentityIdentity1resizing_6/resize/ResizeBilinear:resized_images:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@ъ
NoOpNoOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ё
g
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685

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

g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377

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
ђ

__inference__traced_save_320905
file_prefix@
&read_disablecopyonread_conv2d_9_kernel:4
&read_1_disablecopyonread_conv2d_9_bias:;
(read_2_disablecopyonread_dense_12_kernel:	  4
&read_3_disablecopyonread_dense_12_bias: ;
(read_4_disablecopyonread_dense_13_kernel:	  5
&read_5_disablecopyonread_dense_13_bias:	 L
2read_6_disablecopyonread_conv2d_transpose_9_kernel:>
0read_7_disablecopyonread_conv2d_transpose_9_bias:,
"read_8_disablecopyonread_adam_iter:	 .
$read_9_disablecopyonread_adam_beta_1: /
%read_10_disablecopyonread_adam_beta_2: .
$read_11_disablecopyonread_adam_decay: 6
,read_12_disablecopyonread_adam_learning_rate: +
!read_13_disablecopyonread_total_1: +
!read_14_disablecopyonread_count_1: )
read_15_disablecopyonread_total: )
read_16_disablecopyonread_count: J
0read_17_disablecopyonread_adam_conv2d_9_kernel_m:<
.read_18_disablecopyonread_adam_conv2d_9_bias_m:C
0read_19_disablecopyonread_adam_dense_12_kernel_m:	  <
.read_20_disablecopyonread_adam_dense_12_bias_m: C
0read_21_disablecopyonread_adam_dense_13_kernel_m:	  =
.read_22_disablecopyonread_adam_dense_13_bias_m:	 T
:read_23_disablecopyonread_adam_conv2d_transpose_9_kernel_m:F
8read_24_disablecopyonread_adam_conv2d_transpose_9_bias_m:J
0read_25_disablecopyonread_adam_conv2d_9_kernel_v:<
.read_26_disablecopyonread_adam_conv2d_9_bias_v:C
0read_27_disablecopyonread_adam_dense_12_kernel_v:	  <
.read_28_disablecopyonread_adam_dense_12_bias_v: C
0read_29_disablecopyonread_adam_dense_13_kernel_v:	  =
.read_30_disablecopyonread_adam_dense_13_bias_v:	 T
:read_31_disablecopyonread_adam_conv2d_transpose_9_kernel_v:F
8read_32_disablecopyonread_adam_conv2d_transpose_9_bias_v:
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
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 Њ
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_9_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
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
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_9_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_12_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_12_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
:	  z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_12_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_12_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_13_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_13_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
:	  z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_13_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_13_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnRead2read_6_disablecopyonread_conv2d_transpose_9_kernel"/device:CPU:0*
_output_shapes
 К
Read_6/ReadVariableOpReadVariableOp2read_6_disablecopyonread_conv2d_transpose_9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
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
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_conv2d_transpose_9_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_conv2d_transpose_9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_conv2d_9_kernel_m"/device:CPU:0*
_output_shapes
 К
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_conv2d_9_kernel_m^Read_17/DisableCopyOnRead"/device:CPU:0*&
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
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_conv2d_9_bias_m"/device:CPU:0*
_output_shapes
 Ќ
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_conv2d_9_bias_m^Read_18/DisableCopyOnRead"/device:CPU:0*
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
:
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_dense_12_kernel_m"/device:CPU:0*
_output_shapes
 Г
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_dense_12_kernel_m^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:	  
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_dense_12_bias_m"/device:CPU:0*
_output_shapes
 Ќ
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_dense_12_bias_m^Read_20/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_adam_dense_13_kernel_m"/device:CPU:0*
_output_shapes
 Г
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_adam_dense_13_kernel_m^Read_21/DisableCopyOnRead"/device:CPU:0*
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
:	  
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_dense_13_bias_m"/device:CPU:0*
_output_shapes
 ­
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_dense_13_bias_m^Read_22/DisableCopyOnRead"/device:CPU:0*
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
Read_23/DisableCopyOnReadDisableCopyOnRead:read_23_disablecopyonread_adam_conv2d_transpose_9_kernel_m"/device:CPU:0*
_output_shapes
 Ф
Read_23/ReadVariableOpReadVariableOp:read_23_disablecopyonread_adam_conv2d_transpose_9_kernel_m^Read_23/DisableCopyOnRead"/device:CPU:0*&
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
Read_24/DisableCopyOnReadDisableCopyOnRead8read_24_disablecopyonread_adam_conv2d_transpose_9_bias_m"/device:CPU:0*
_output_shapes
 Ж
Read_24/ReadVariableOpReadVariableOp8read_24_disablecopyonread_adam_conv2d_transpose_9_bias_m^Read_24/DisableCopyOnRead"/device:CPU:0*
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
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_conv2d_9_kernel_v"/device:CPU:0*
_output_shapes
 К
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_conv2d_9_kernel_v^Read_25/DisableCopyOnRead"/device:CPU:0*&
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
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_conv2d_9_bias_v"/device:CPU:0*
_output_shapes
 Ќ
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_conv2d_9_bias_v^Read_26/DisableCopyOnRead"/device:CPU:0*
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
:
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_dense_12_kernel_v"/device:CPU:0*
_output_shapes
 Г
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_dense_12_kernel_v^Read_27/DisableCopyOnRead"/device:CPU:0*
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
:	  
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_dense_12_bias_v"/device:CPU:0*
_output_shapes
 Ќ
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_dense_12_bias_v^Read_28/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_dense_13_kernel_v"/device:CPU:0*
_output_shapes
 Г
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_dense_13_kernel_v^Read_29/DisableCopyOnRead"/device:CPU:0*
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
:	  
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_dense_13_bias_v"/device:CPU:0*
_output_shapes
 ­
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_dense_13_bias_v^Read_30/DisableCopyOnRead"/device:CPU:0*
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
Read_31/DisableCopyOnReadDisableCopyOnRead:read_31_disablecopyonread_adam_conv2d_transpose_9_kernel_v"/device:CPU:0*
_output_shapes
 Ф
Read_31/ReadVariableOpReadVariableOp:read_31_disablecopyonread_adam_conv2d_transpose_9_kernel_v^Read_31/DisableCopyOnRead"/device:CPU:0*&
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
Read_32/DisableCopyOnReadDisableCopyOnRead8read_32_disablecopyonread_adam_conv2d_transpose_9_bias_v"/device:CPU:0*
_output_shapes
 Ж
Read_32/ReadVariableOpReadVariableOp8read_32_disablecopyonread_adam_conv2d_transpose_9_bias_v^Read_32/DisableCopyOnRead"/device:CPU:0*
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
ь
д
C__inference_Decoder_layer_call_and_return_conditional_losses_319780

inputs"
dense_13_319766:	  
dense_13_319768:	 3
conv2d_transpose_9_319772:'
conv2d_transpose_9_319774:
identityЂ*conv2d_transpose_9/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallё
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_319766dense_13_319768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_319706х
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726М
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_transpose_9_319772conv2d_transpose_9_319774*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662
up_sampling2d_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685ч
resizing_6/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740{
IdentityIdentity#resizing_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_9/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 
Х
C__inference_Encoder_layer_call_and_return_conditional_losses_319506

inputs)
conv2d_9_319484:
conv2d_9_319486:"
dense_12_319491:	  
dense_12_319493: 
identity

identity_1Ђ conv2d_9/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallј
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_319484conv2d_9_319486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377н
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_319491dense_12_319493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437Ъ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8 *9
f4R2
0__inference_dense_12_activity_regularizer_319396
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ў
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ц
H__inference_dense_12_layer_call_and_return_all_conditional_losses_320559

inputs
unknown:	  
	unknown_0: 
identity

identity_1ЂStatefulPartitionedCallй
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437Є
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
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_12_activity_regularizer_319396o
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
§
G
0__inference_dense_12_activity_regularizer_319396
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
ь
д
C__inference_Decoder_layer_call_and_return_conditional_losses_319810

inputs"
dense_13_319796:	  
dense_13_319798:	 3
conv2d_transpose_9_319802:'
conv2d_transpose_9_319804:
identityЂ*conv2d_transpose_9/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallё
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_319796dense_13_319798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_319706х
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726М
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_transpose_9_319802conv2d_transpose_9_319804*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662
up_sampling2d_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685ч
resizing_6/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740{
IdentityIdentity#resizing_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_9/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
И
L
0__inference_max_pooling2d_9_layer_call_fn_320523

inputs
identityй
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
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377
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
И
ж
(__inference_Encoder_layer_call_fn_319518
input_19!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319506o
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
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_19
Ё
g
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_320673

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
В
д
(__inference_Encoder_layer_call_fn_320296

inputs!
unknown:
	unknown_0:
	unknown_1:	  
	unknown_2: 
identityЂStatefulPartitionedCallѕ
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
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319506o
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
Ђ 
Ч
C__inference_Encoder_layer_call_and_return_conditional_losses_319453
input_19)
conv2d_9_319412:
conv2d_9_319414:"
dense_12_319438:	  
dense_12_319440: 
identity

identity_1Ђ conv2d_9/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallњ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_19conv2d_9_319412conv2d_9_319414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377н
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_319438dense_12_319440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437Ъ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8 *9
f4R2
0__inference_dense_12_activity_regularizer_319396
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ў
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_19
 
Х
C__inference_Encoder_layer_call_and_return_conditional_losses_319545

inputs)
conv2d_9_319523:
conv2d_9_319525:"
dense_12_319530:	  
dense_12_319532: 
identity

identity_1Ђ conv2d_9/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallј
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_319523conv2d_9_319525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377н
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_319530dense_12_319532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437Ъ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8 *9
f4R2
0__inference_dense_12_activity_regularizer_319396
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ў
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л'
ч
C__inference_Encoder_layer_call_and_return_conditional_losses_320378

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource::
'dense_12_matmul_readvariableop_resource:	  6
(dense_12_biasadd_readvariableop_resource: 
identity

identity_1Ђconv2d_9/BiasAdd/ReadVariableOpЂconv2d_9/Conv2D/ReadVariableOpЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOp
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ќ
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ**
paddingVALID*
strides

conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*h
conv2d_9/EluEluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*Њ
max_pooling2d_9/MaxPoolMaxPoolconv2d_9/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_6/ReshapeReshape max_pooling2d_9/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_12/EluEludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ u
 dense_12/ActivityRegularizer/AbsAbsdense_12/Elu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ s
"dense_12/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_12/ActivityRegularizer/SumSum$dense_12/ActivityRegularizer/Abs:y:0+dense_12/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_12/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8 
 dense_12/ActivityRegularizer/mulMul+dense_12/ActivityRegularizer/mul/x:output:0)dense_12/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: z
"dense_12/ActivityRegularizer/ShapeShapedense_12/Elu:activations:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_12/ActivityRegularizer/truedivRealDiv$dense_12/ActivityRegularizer/mul:z:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: i
IdentityIdentitydense_12/Elu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ь
NoOpNoOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ђ
ж
C__inference_Decoder_layer_call_and_return_conditional_losses_319743
input_20"
dense_13_319707:	  
dense_13_319709:	 3
conv2d_transpose_9_319728:'
conv2d_transpose_9_319730:
identityЂ*conv2d_transpose_9/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallѓ
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_13_319707dense_13_319709*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_319706х
reshape_6/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_6_layer_call_and_return_conditional_losses_319726М
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_transpose_9_319728conv2d_transpose_9_319730*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662
up_sampling2d_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_319685ч
resizing_6/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_resizing_6_layer_call_and_return_conditional_losses_319740{
IdentityIdentity#resizing_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@
NoOpNoOp+^conv2d_transpose_9/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ : : : : 2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
input_20
Ю
b
F__inference_resizing_6_layer_call_and_return_conditional_losses_320684

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
Ђ 
Ч
C__inference_Encoder_layer_call_and_return_conditional_losses_319478
input_19)
conv2d_9_319456:
conv2d_9_319458:"
dense_12_319463:	  
dense_12_319465: 
identity

identity_1Ђ conv2d_9/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallњ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_19conv2d_9_319456conv2d_9_319458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_319411ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_319377н
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_319424
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_319463dense_12_319465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_319437Ъ
,dense_12/ActivityRegularizer/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8 *9
f4R2
0__inference_dense_12_activity_regularizer_319396
"dense_12/ActivityRegularizer/ShapeShape)dense_12/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::эЯz
0dense_12/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_12/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_12/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*dense_12/ActivityRegularizer/strided_sliceStridedSlice+dense_12/ActivityRegularizer/Shape:output:09dense_12/ActivityRegularizer/strided_slice/stack:output:0;dense_12/ActivityRegularizer/strided_slice/stack_1:output:0;dense_12/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_12/ActivityRegularizer/CastCast3dense_12/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ў
$dense_12/ActivityRegularizer/truedivRealDiv5dense_12/ActivityRegularizer/PartitionedCall:output:0%dense_12/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ h

Identity_1Identity(dense_12/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^conv2d_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ@: : : : 2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_19
С
н
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319894
input_21(
encoder_319873:
encoder_319875:!
encoder_319877:	  
encoder_319879: !
decoder_319883:	  
decoder_319885:	 (
decoder_319887:
decoder_319889:
identity

identity_1ЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_21encoder_319873encoder_319875encoder_319877encoder_319879*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_319506Л
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_319883decoder_319885decoder_319887decoder_319889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_319780
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
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
input_21

g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_320528

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
њ#

N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_319662

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
Ц

)__inference_dense_13_layer_call_fn_320579

inputs
unknown:	  
	unknown_0:	 
identityЂStatefulPartitionedCallк
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_319706p
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
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
F
input_21:
serving_default_input_21:0џџџџџџџџџ@D
Decoder9
StatefulPartitionedCall:0џџџџџџџџџ@tensorflow/serving/predict:іЛ
О
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
signatures"
_tf_keras_network
"
_tf_keras_input_layer

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_sequential
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object

/trace_0
0trace_1
1trace_2
2trace_32Ј
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_319965
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320011
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320106
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320128Е
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
 z/trace_0z0trace_1z1trace_2z2trace_3
џ
3trace_0
4trace_1
5trace_2
6trace_32
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319894
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319918
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320205
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320282Е
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
ЭBЪ
!__inference__wrapped_model_319371input_21"
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
7iter

8beta_1

9beta_2
	:decay
;learning_rate"mн#mо$mп%mр&mс'mт(mу)mф"vх#vц$vч%vш&vщ'vъ(vы)vь"
	optimizer
,
<serving_default"
signature_map
н
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

"kernel
#bias
 C_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
<
"0
#1
$2
%3"
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ы
[trace_0
\trace_1
]trace_2
^trace_32р
(__inference_Encoder_layer_call_fn_319518
(__inference_Encoder_layer_call_fn_319557
(__inference_Encoder_layer_call_fn_320296
(__inference_Encoder_layer_call_fn_320310Е
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
 z[trace_0z\trace_1z]trace_2z^trace_3
З
_trace_0
`trace_1
atrace_2
btrace_32Ь
C__inference_Encoder_layer_call_and_return_conditional_losses_319453
C__inference_Encoder_layer_call_and_return_conditional_losses_319478
C__inference_Encoder_layer_call_and_return_conditional_losses_320344
C__inference_Encoder_layer_call_and_return_conditional_losses_320378Е
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
 z_trace_0z`trace_1zatrace_2zbtrace_3
Л
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
Ѕ
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
н
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

(kernel
)bias
 u_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
г
trace_0
trace_1
trace_2
trace_32р
(__inference_Decoder_layer_call_fn_319791
(__inference_Decoder_layer_call_fn_319821
(__inference_Decoder_layer_call_fn_320391
(__inference_Decoder_layer_call_fn_320404Е
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
 ztrace_0ztrace_1ztrace_2ztrace_3
П
trace_0
trace_1
trace_2
trace_32Ь
C__inference_Decoder_layer_call_and_return_conditional_losses_319743
C__inference_Decoder_layer_call_and_return_conditional_losses_319760
C__inference_Decoder_layer_call_and_return_conditional_losses_320451
C__inference_Decoder_layer_call_and_return_conditional_losses_320498Е
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
 ztrace_0ztrace_1ztrace_2ztrace_3
):'2conv2d_9/kernel
:2conv2d_9/bias
": 	  2dense_12/kernel
: 2dense_12/bias
": 	  2dense_13/kernel
: 2dense_13/bias
3:12conv2d_transpose_9/kernel
%:#2conv2d_transpose_9/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_319965input_21"Е
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
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320011input_21"Е
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
Bў
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320106inputs"Е
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
Bў
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320128inputs"Е
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
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319894input_21"Е
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
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319918input_21"Е
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
B
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320205inputs"Е
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
B
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320282inputs"Е
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
$__inference_signature_wrapper_320084input_21"
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
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_conv2d_9_layer_call_fn_320507
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
 ztrace_0

trace_02с
D__inference_conv2d_9_layer_call_and_return_conditional_losses_320518
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
 ztrace_0
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ь
trace_02Э
0__inference_max_pooling2d_9_layer_call_fn_320523
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
 ztrace_0

trace_02ш
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_320528
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
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ц
Єtrace_02Ч
*__inference_flatten_6_layer_call_fn_320533
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

Ѕtrace_02т
E__inference_flatten_6_layer_call_and_return_conditional_losses_320539
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
 zЅtrace_0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
Ћactivity_regularizer_fn
*U&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
х
­trace_02Ц
)__inference_dense_12_layer_call_fn_320548
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
 z­trace_0

Ўtrace_02х
H__inference_dense_12_layer_call_and_return_all_conditional_losses_320559
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
 zЎtrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
(__inference_Encoder_layer_call_fn_319518input_19"Е
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
(__inference_Encoder_layer_call_fn_319557input_19"Е
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
яBь
(__inference_Encoder_layer_call_fn_320296inputs"Е
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
яBь
(__inference_Encoder_layer_call_fn_320310inputs"Е
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
C__inference_Encoder_layer_call_and_return_conditional_losses_319453input_19"Е
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
C__inference_Encoder_layer_call_and_return_conditional_losses_319478input_19"Е
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
B
C__inference_Encoder_layer_call_and_return_conditional_losses_320344inputs"Е
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
B
C__inference_Encoder_layer_call_and_return_conditional_losses_320378inputs"Е
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
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
х
Дtrace_02Ц
)__inference_dense_13_layer_call_fn_320579
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
 zДtrace_0

Еtrace_02с
D__inference_dense_13_layer_call_and_return_conditional_losses_320590
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
 zЕtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
ц
Лtrace_02Ч
*__inference_reshape_6_layer_call_fn_320595
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

Мtrace_02т
E__inference_reshape_6_layer_call_and_return_conditional_losses_320609
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
 zМtrace_0
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
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
я
Тtrace_02а
3__inference_conv2d_transpose_9_layer_call_fn_320618
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

Уtrace_02ы
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_320656
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
 zУtrace_0
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
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
ь
Щtrace_02Э
0__inference_up_sampling2d_9_layer_call_fn_320661
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

Ъtrace_02ш
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_320673
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
 zЪtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ч
аtrace_02Ш
+__inference_resizing_6_layer_call_fn_320678
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

бtrace_02у
F__inference_resizing_6_layer_call_and_return_conditional_losses_320684
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
 zбtrace_0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
(__inference_Decoder_layer_call_fn_319791input_20"Е
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
(__inference_Decoder_layer_call_fn_319821input_20"Е
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
яBь
(__inference_Decoder_layer_call_fn_320391inputs"Е
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
яBь
(__inference_Decoder_layer_call_fn_320404inputs"Е
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
C__inference_Decoder_layer_call_and_return_conditional_losses_319743input_20"Е
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
C__inference_Decoder_layer_call_and_return_conditional_losses_319760input_20"Е
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
B
C__inference_Decoder_layer_call_and_return_conditional_losses_320451inputs"Е
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
B
C__inference_Decoder_layer_call_and_return_conditional_losses_320498inputs"Е
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
в	variables
г	keras_api

дtotal

еcount"
_tf_keras_metric
c
ж	variables
з	keras_api

иtotal

йcount
к
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
гBа
)__inference_conv2d_9_layer_call_fn_320507inputs"
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
D__inference_conv2d_9_layer_call_and_return_conditional_losses_320518inputs"
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
кBз
0__inference_max_pooling2d_9_layer_call_fn_320523inputs"
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
ѕBђ
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_320528inputs"
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
дBб
*__inference_flatten_6_layer_call_fn_320533inputs"
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
E__inference_flatten_6_layer_call_and_return_conditional_losses_320539inputs"
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
лtrace_02д
0__inference_dense_12_activity_regularizer_319396
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
	zлtrace_0

мtrace_02с
D__inference_dense_12_layer_call_and_return_conditional_losses_320570
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
 zмtrace_0
гBа
)__inference_dense_12_layer_call_fn_320548inputs"
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
H__inference_dense_12_layer_call_and_return_all_conditional_losses_320559inputs"
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
)__inference_dense_13_layer_call_fn_320579inputs"
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
D__inference_dense_13_layer_call_and_return_conditional_losses_320590inputs"
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
дBб
*__inference_reshape_6_layer_call_fn_320595inputs"
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
E__inference_reshape_6_layer_call_and_return_conditional_losses_320609inputs"
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
нBк
3__inference_conv2d_transpose_9_layer_call_fn_320618inputs"
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
јBѕ
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_320656inputs"
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
кBз
0__inference_up_sampling2d_9_layer_call_fn_320661inputs"
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
ѕBђ
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_320673inputs"
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
+__inference_resizing_6_layer_call_fn_320678inputs"
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
F__inference_resizing_6_layer_call_and_return_conditional_losses_320684inputs"
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
д0
е1"
trackable_list_wrapper
.
в	variables"
_generic_user_object
:  (2total
:  (2count
0
и0
й1"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
мBй
0__inference_dense_12_activity_regularizer_319396x"
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
D__inference_dense_12_layer_call_and_return_conditional_losses_320570inputs"
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
.:,2Adam/conv2d_9/kernel/m
 :2Adam/conv2d_9/bias/m
':%	  2Adam/dense_12/kernel/m
 : 2Adam/dense_12/bias/m
':%	  2Adam/dense_13/kernel/m
!: 2Adam/dense_13/bias/m
8:62 Adam/conv2d_transpose_9/kernel/m
*:(2Adam/conv2d_transpose_9/bias/m
.:,2Adam/conv2d_9/kernel/v
 :2Adam/conv2d_9/bias/v
':%	  2Adam/dense_12/kernel/v
 : 2Adam/dense_12/bias/v
':%	  2Adam/dense_13/kernel/v
!: 2Adam/dense_13/bias/v
8:62 Adam/conv2d_transpose_9/kernel/v
*:(2Adam/conv2d_transpose_9/bias/vє
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319894"#$%&'()BЂ?
8Ђ5
+(
input_21џџџџџџџџџ@
p

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 є
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_319918"#$%&'()BЂ?
8Ђ5
+(
input_21џџџџџџџџџ@
p 

 
Њ "JЂG
+(
tensor_0џџџџџџџџџ@



tensor_1_0 ђ
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320205"#$%&'()@Ђ=
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

tensor_1_0 ђ
U__inference_AE_Conv_prep_flatten_STFT_layer_call_and_return_conditional_losses_320282"#$%&'()@Ђ=
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
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_319965z"#$%&'()BЂ?
8Ђ5
+(
input_21џџџџџџџџџ@
p

 
Њ "*'
unknownџџџџџџџџџ@И
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320011z"#$%&'()BЂ?
8Ђ5
+(
input_21џџџџџџџџџ@
p 

 
Њ "*'
unknownџџџџџџџџџ@Ж
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320106x"#$%&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "*'
unknownџџџџџџџџџ@Ж
:__inference_AE_Conv_prep_flatten_STFT_layer_call_fn_320128x"#$%&'()@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "*'
unknownџџџџџџџџџ@П
C__inference_Decoder_layer_call_and_return_conditional_losses_319743x&'()9Ђ6
/Ђ,
"
input_20џџџџџџџџџ 
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 П
C__inference_Decoder_layer_call_and_return_conditional_losses_319760x&'()9Ђ6
/Ђ,
"
input_20џџџџџџџџџ 
p 

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 Н
C__inference_Decoder_layer_call_and_return_conditional_losses_320451v&'()7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 Н
C__inference_Decoder_layer_call_and_return_conditional_losses_320498v&'()7Ђ4
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
(__inference_Decoder_layer_call_fn_319791m&'()9Ђ6
/Ђ,
"
input_20џџџџџџџџџ 
p

 
Њ "*'
unknownџџџџџџџџџ@
(__inference_Decoder_layer_call_fn_319821m&'()9Ђ6
/Ђ,
"
input_20џџџџџџџџџ 
p 

 
Њ "*'
unknownџџџџџџџџџ@
(__inference_Decoder_layer_call_fn_320391k&'()7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "*'
unknownџџџџџџџџџ@
(__inference_Decoder_layer_call_fn_320404k&'()7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p 

 
Њ "*'
unknownџџџџџџџџџ@е
C__inference_Encoder_layer_call_and_return_conditional_losses_319453"#$%BЂ?
8Ђ5
+(
input_19џџџџџџџџџ@
p

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 е
C__inference_Encoder_layer_call_and_return_conditional_losses_319478"#$%BЂ?
8Ђ5
+(
input_19џџџџџџџџџ@
p 

 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 г
C__inference_Encoder_layer_call_and_return_conditional_losses_320344"#$%@Ђ=
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

tensor_1_0 г
C__inference_Encoder_layer_call_and_return_conditional_losses_320378"#$%@Ђ=
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
(__inference_Encoder_layer_call_fn_319518m"#$%BЂ?
8Ђ5
+(
input_19џџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ 
(__inference_Encoder_layer_call_fn_319557m"#$%BЂ?
8Ђ5
+(
input_19џџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ 
(__inference_Encoder_layer_call_fn_320296k"#$%@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ 
(__inference_Encoder_layer_call_fn_320310k"#$%@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ Ј
!__inference__wrapped_model_319371"#$%&'():Ђ7
0Ђ-
+(
input_21џџџџџџџџџ@
Њ ":Њ7
5
Decoder*'
decoderџџџџџџџџџ@М
D__inference_conv2d_9_layer_call_and_return_conditional_losses_320518t"#8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ*
 
)__inference_conv2d_9_layer_call_fn_320507i"#8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ@
Њ ")&
unknownџџџџџџџџџ*ъ
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_320656()IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
3__inference_conv2d_transpose_9_layer_call_fn_320618()IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџc
0__inference_dense_12_activity_regularizer_319396/Ђ
Ђ
	
x
Њ "
unknown Х
H__inference_dense_12_layer_call_and_return_all_conditional_losses_320559y$%0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "AЂ>
"
tensor_0џџџџџџџџџ 



tensor_1_0 Ќ
D__inference_dense_12_layer_call_and_return_conditional_losses_320570d$%0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
)__inference_dense_12_layer_call_fn_320548Y$%0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџ Ќ
D__inference_dense_13_layer_call_and_return_conditional_losses_320590d&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
)__inference_dense_13_layer_call_fn_320579Y&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџ Б
E__inference_flatten_6_layer_call_and_return_conditional_losses_320539h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
*__inference_flatten_6_layer_call_fn_320533]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ ѕ
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_320528ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
0__inference_max_pooling2d_9_layer_call_fn_320523RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџБ
E__inference_reshape_6_layer_call_and_return_conditional_losses_320609h0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
*__inference_reshape_6_layer_call_fn_320595]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџЭ
F__inference_resizing_6_layer_call_and_return_conditional_losses_320684IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@
 І
+__inference_resizing_6_layer_call_fn_320678wIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџ@З
$__inference_signature_wrapper_320084"#$%&'()FЂC
Ђ 
<Њ9
7
input_21+(
input_21џџџџџџџџџ@":Њ7
5
Decoder*'
decoderџџџџџџџџџ@ѕ
K__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_320673ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
0__inference_up_sampling2d_9_layer_call_fn_320661RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ