(*
uvolume
Copyright (C) 2016 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

// IMPORTANT: uncomment or define either AVX, AVX2 or AVX512
// depending on your processor (AVX512 is experimental at this point).
//{$DEFINE AVX}

// AVX2 setting depends on your CPU. Enable it if it allows vfmadd231ps only.
//{$DEFINE AVX2}

{$IFDEF FPC}
  {$COPERATORS ON}
  {$mode objfpc}
  {$INLINE ON}
  {$MACRO ON}
  {$LONGSTRINGS ON}
  {$CODEPAGE UTF8}
{$ELSE}
  // AVX code is supported only under FPC
  {$UNDEF AVX}
  {$UNDEF AVX2}
  {$UNDEF AVX512}
  {$UNDEF AVXANY}

  {$IFDEF CPU32BITS}
    {$DEFINE CPU32}
  {$ENDIF}

  {$IFDEF CPU64BITS}
    {$DEFINE CPU64}
  {$ENDIF}
{$ENDIF}

{$IFDEF CPU32}
  {$IFDEF AVX}
    {$DEFINE AVX32}
  {$ENDIF}
  {$IFDEF AVX2}
    {$DEFINE AVX32}
  {$ENDIF}
{$ENDIF}

{$IFDEF CPU64}
  {$IFDEF AVX}
    {$DEFINE AVX64}
  {$ENDIF}
  {$IFDEF AVX2}
    {$DEFINE AVX64}
  {$ENDIF}
{$ENDIF}

{$IFDEF AVX512}
{$DEFINE AVX64}
{$UNDEF AVX32}
{$ENDIF}

{$IFDEF AVX32}
{$DEFINE AVXANY}
{$ENDIF}

{$IFDEF AVX64}
{$DEFINE AVXANY}
{$ENDIF}

{$IFDEF AVXANY}
{$include neuralasm.inc}
{$ENDIF}
