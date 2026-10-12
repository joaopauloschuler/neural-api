unit U_PrimesFromDigits_mX4_Experimental476;

//P:array of integer; of TIntPartition= class(Tobject) is missing!
//https://medium.com/@mohankrishnagupta/python-best-practices-4ad47c81b9bc

interface

(*uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls,
  strutils, ShellAPI, ExtCtrls {manually added to access PosEx function};  *)

type
  TForm1 = {class(}TForm;
  var
    Panel1: TPanel;
    aMemo1: TMemo;
    Part1Btn: TButton;
    Part2Btn: TButton;
    aMemo2: TMemo;
    aMemo3: TMemo;
    image,image1,image2,image3,image4,sourceimage:TImage; JpegImage:TJpegImage; 
    StaticText1: TStaticText;
    procedure TForm1FormActivate(Sender: TObject);
    procedure TForm1Part1BtnClick(Sender: TObject);
    procedure TForm1Part2BtnClick(Sender: TObject);
    procedure TForm1Memo2Click(Sender: TObject);
    procedure TForm1StaticText1Click(Sender: TObject);
    
  //public
    { Public declarations }
    var
    count:integer;
    baseparts:array of string;
    basepartsindex: array of integer; {pointers to base partition}
    parts:array of array of integer;
    //partscounts:array of integer;
    list:TStringlist;
    combos: TComboSet; primes: TPrimes; defpartition: TIntPartition;    
    basepartscount:integer;
    //function Callback:boolean;
    function NextPartitionCallback:boolean;  {receives partitions one per call}
    procedure FindPrimeSets(index:integer; list:TStrings; var maxtoReturn:integer);
  //end;

var
 Form1: TForm1;

implementation

//{$R *.DFM}

//uses mathslib, ucombov2, uintegerpartition2;

procedure TForm1FormActivate(Sender: TObject);
begin
  list:=TStringlist.create;
  list.sorted:=true;
  combos:= TComboSet.create;
  primes:= TPrimes.create;
  defpartition:= Tintpartition.create;
end;

{ CL.AddTypeS('TCombotype', '( ucombinations, uPermutations, CombinationsDown, Pe'
   +'rmutationsDown, CombinationsCoLex, CombinationsCoLexDown, PermutationsRepe'
   +'at, PermutationsWithRep, PermutationsRepeatDown, CombinationsWithrep, Comb'
   +'inationsRepeat, CombinationsRepeatDown )');}
 //http://www.delphibasics.co.uk/Article.asp?Name=Sets  

{*********** Part1BtnClick ************}
procedure TForm1Part1BtnClick(Sender: TObject);
var
  i,primecount:integer;
  p:int64;
  act: TCombotype;
  aSelected: TByteArray64;
begin
  act:= upermutations; //CombinationsWithrep; //ord(TCombotype(''));  
  writeln('Comb Type is: '+intToStr(ord(act)));
  //act:= TCombotype(ord(CombinationsDown)); //ord(TCombotype(''));
   //act:= 2;                            
  primecount:=0;                                     
  //combos.setup(9,9,TCombotype(ord(act))); 
  combos.setup(9,9,upermutations);
  writeln(itoa(length(combos.selected)));
  //for it:= 1 to length(combos.selected)-1 do 
    //amemo2.lines.add(itoa(aselected[it]));
  with combos  do
  while getnext do
   aselected:= selected;
   //amemo2.lines.add(itoa(aselected[i]));                
  //if aselected[9] in [1,3,7,9] then
   amemo2.lines.add(itoa(aselected[2])); 
   if aselected[9] = (1 or 3 or 7 or 9) then begin
     p:=aselected[1];
     amemo2.lines.add(itoa(p));
     for i:=2 to 9 do begin 
        p:=10*p + aselected[i];
        amemo2.lines.add(itoa(aselected[i])); 
     end;    
     if primes.isprime(p) then begin
      inc(primecount);
     end;
   end;
  //amemo2.Clear;
  showmessage(format('There are %d primes containing exactly one occurrence of the digits 1 through 9.'
                  +#13+'No matter how they are arranged, the sum of digits 1 though 9 is 45, a multiple of 3.'
                  +#13+'Any number whose digits sum to a multiple of 3 is divisible by 3. i.e. is not prime',
                  [primecount]));
                  
end;

{************* NextPartitionCallback **********}
function NextPartitionCallback:boolean;
var
  i:integer;
  s:string;
  P1:array of integer;
begin
  result:=true;
  with defpartition do
  begin
    {Save partition definitions for later use}
    {we'll save all unique permutations of this partition}
    s:='';
    p1:= p;
    for i:=0 to partsize-1 do s:=s+','+inttostr(p1[partsize-1-i]);
    inc(count);
    inc(basepartscount);
    delete(s,1,1);
    baseparts[basepartscount]:=s;
    if count>=length(parts) then
    begin
      setlength(parts,count+100);
      setlength(basepartsindex,count+100);
    end;
    basepartsindex[count]:=basepartscount;
    setlength(parts[count],partsize+1);
    for i:=1 to partsize do parts[count][i]:=p1[partsize-i];
    parts[count][0]:=0;
  end;
  setlength(parts,count+1);
end;


{************** Part2BtnClick *************}
procedure TForm1Part2BtnClick(Sender: TObject);
var
  N:integer;
  i,j,k:integer;
  startat:integer;
  s:string;
  p:integer;
  ok:boolean;
  pp:array[1..9] of integer;
  pcount:integer;
  temp,index:integer;
  basepartscounts:array of integer;
  totpartscount, dupcount:integer;
  starttime:TDatetime; aSelected: TByteArray64;
begin
  amemo2.Clear;
  screen.Cursor:=crHourglass;
  n:=defpartition.partitioncount(9,0);
  setlength(parts,n+1);
  setlength(baseparts,n+1);
  setlength(basepartscounts,500);
  setlength(basepartsindex,n+1);
  count:=0;
  dupcount:=0;
  basepartscount:=0;
  totpartscount:=0;
  //defpartition.PartitionInit(9,0,NextPartitionCallback);
  {now we have generated and saved all paritions,
   generate permutations of 123456789 and break into
   a set of numbers based on each partition.  then check
   each number in the set for "primeness"}
  list.clear;
  starttime:=now;

  combos.setup(9,9,upermutations);
   
  with combos  do
  while getnext do
  aselected:= selected; 
  //if aselected[9] in [1,3,7,9] then begin
  if aselected[9] = (1 or 3 or 7 or 9) then begin
    for i:=1 to count do
    begin
      ok:=true;
      startat:=1;
      pcount:=0;
      s:='';
      for j:=1 to high(parts[i]) do
      begin {split this permutation into primes based on }
        p:=0;
        for k:=startat to startat+parts[i][j]-1  do
        p:=10*p+aselected[k];

        if not primes.isprime(p) then
        begin
          ok:=false;
          break;
        end;
        inc(pcount);
        pp[pcount]:=p;
        startat:=startat+parts[i][j];
      end;
      if OK then
      begin
        {sort this set of primes from smallest to largest value}
        for j:=1 to pcount-1 do
        for k:= j+1 to pcount do
        if pp[j]>pp[k] then
        begin
          temp:=pp[j];
          pp[j]:=pp[k];
          pp[k]:=temp;
        end;
        s:='';
        for j:=1 to pcount do s:=s+','+inttostr(pp[j]);
        delete(s,1,1);
        if not list.find(s,index) then
        begin {some sets may be duplicates, only save the first of those}
          list.add(s);
          (*
          if baseparts[basepartsindex[i]]='1,2,2,2,2'
          then memo2.lines.add(format('Sets of primes with lengths %s: %s',
                 [baseparts[basepartsindex[i]],s]));
          *)
          inc(totpartscount);
          inc(parts[i][0]);  {Keep counts by partition}
        end
        else inc(dupcount);
      end;
    end;
  end;

  temp:=0;
  for i:=1 to count do
  begin
    s:='';
    for j:=1 to high(parts[i]) do s:=s+','+inttostr(parts[i][j]);
    delete(s,1,1);
    memo2.lines.add(format('[%s] has %d prime sets',
                           [s,parts[i][0]{,baseparts[basepartsindex[i]]}]));
    inc2(temp, parts[i][0]);
  end;

  (*
  {for debugging - add up the individual partition counts in "temp" to compare with total count}
  temp:=0;
  for i:= 1 to basepartscount do
  begin
    memo2.lines.add(format('For partition  [%s], there are %d prime sets',[baseparts[i],basepartscounts[i]]));
    inc(temp,basepartscounts[i]);
  end;
  *)

  screen.Cursor:=crdefault;
  If totpartscount<>temp
  then showmessage(format('Program error! Detail count(%.0n) <> Total count (%.0n)',
                             [0.0+temp,(now-starttime)*secsperday, 0.0+totpartscount]))
  else

  showmessage(format('%.0n unique prime sets found in %.1f seconds'
                   +#13+'%d duplicate prime sets found',
                   [0.0+totpartscount,(now-starttime)*secsperday, dupcount]));

end;


function LineClicked(Memo:TMemo):integer;
{For a click on a memo line return the line number (number is relative to 0)}
begin
  with memo do result:=Perform(Em_LineFromChar,Selstart,0);
end;


function LinePositionClicked(Memo:TMemo):integer;
{When a TMemo line is clicked, return the character position
 within the line (position is relative to 1)}
var LineIndex:integer;
begin
  with memo do begin
     LineIndex:=Perform(Em_LineIndex,Lineclicked(memo),0);
     Result:=Selstart-Lineindex+1;
  end;
end;


{************ FindPrimeSets ***************}
procedure FindPrimeSets(index:integer; list:TStrings; var maxtoreturn:Integer);
{Performs the same search as the original Part2 button except does it
 for only the particular partitioning licked by the user}
var
  j,k:integer;
  ok:boolean;
  startat,p,pcount,temp:integer;
  pp:array[1..9] of integer; {primes found for a particular arrangement of digits}
  s:string;
  dummy:integer;  aSelected: TByteArray64;
begin
  combos.setup(9,9,upermutations);
  with combos  do
  while getnext do
  aSelected:= selected;
  if aselected[9] in [1,3,7,9] then begin
    ok:=true;
    startat:=1;
    pcount:=0;
    s:='';
    for j:=1 to high(parts[index]) do
    begin {split this permutation into primes based on }
      p:=0;
      for k:=startat to startat+parts[index][j]-1  do
      p:=10*p+aselected[k];

      if not primes.isprime(p) then
      begin
        ok:=false;
        break;
      end;
      inc(pcount);
      pp[pcount]:=p;
      startat:=startat+parts[index][j];
    end;
    if OK then begin

      {sort this set of primes from smallest to largest value}
      {makes is easy to delete duplicate prime sets}
      for j:=1 to pcount-1 do begin
      for k:= j+1 to pcount do
      if pp[j]>pp[k] then begin
        temp:=pp[j];
        pp[j]:=pp[k];
        pp[k]:=temp;
      end;

      s:='';
      for j:=1 to pcount do s:=s+','+inttostr(pp[j]);
      delete(s,1,1);
      {some sets may be duplicates, only save the first of those}
      dummy:=list.IndexOf(s);
      if dummy<0 then list.add(s);
      if list.count>=maxtoreturn then break;
    end;
  end;
  end;
  maxtoreturn:=list.Count;
end;

{**************** Memo2Click ************}
procedure TForm1Memo2Click(Sender: TObject);
var
  lineNbr:integer;
  i,n,n2:integer;
  line,partition:string;
  maxtoreturn:integer;
begin
  LineNbr:=lineclicked(aMemo2);
  line:=amemo2.lines[lineNbr];
  n:=pos('[',line);
  if n>0 then
  begin
    n2:=posex(']',line,n+1);
    partition:=copy(line,N+1, n2-n-1);
    if baseparts[linenbr+1]=partition then
    begin
      amemo3.clear;
      maxtoreturn:=100;
      FindPrimeSets(linenbr+1, amemo3.lines, maxtoreturn);
    end
    else showmessage(format('Program error: Partition %s not found',[partition]));
  end
  else Showmessage('No partition found on clicked line');
end;

const
  MAXX = 40 ;
type
  ImgObj = record
    Addrs : string;
    X: Integer ;
    Y: Integer ;
  end;

var
  All : array[1..MAXX] of ImgObj ;

procedure TForm1btn1Clickcanvas(Sender: TObject);
var
  BuffBitmap :TBitmap ;
  I,j,k: Integer;
begin
  // set  all bit maps
  //....
  // draw 40 images
  BuffBitmap := TBitmap.Create ;
  for I := 1 to MAXX do begin
    BuffBitmap.LoadFromFile(All[i].Addrs);
    for j := 0 to BuffBitmap.Width-1 do
      for k := 0 to BuffBitmap.Height-1 do 
       Self.Canvas.Pixels[All[i].X+j,All[i].Y+k]:=BuffBitmap.Canvas.Pixels[j,k];
  end;
  BuffBitmap.free;
end;

//https://thecodecave.com/treating-a-timages-loaded-jpeg-as-a-bitmap-in-delphi/
procedure TForm5Image1Click(Sender: TObject; image1: TImage);
var
  RS: TResourceStream;
  MS: TMemoryStream;
  JPGImage: TJPEGImage;  var AName :String;
begin
  JPGImage := TJPEGImage.Create;
  try
   aname:= 'MAXBOX4LOGO';
    RS:= TResourceStream.Create(hInstance, aname, 'JPEG');        
    //MS:= TMemoryStream.create;
    //GetResourceName( ObjStream : TStream; var AName :Str) :Bool 
    //writeln('resstream '+botostr(GetResourceName( MS, aname))); 
    //LoadGraphicFromResource(Graphic:TGraphic; const ResName:str; ResType:PChar);
    //LoadGraphicFromResource(Image1.Picture.Graphic, aname, 'RT_RCDATA');    
    //LoadGraphicFromResource(Image1.Picture.Graphic, aname, 'JPEG');    
    try
      JPGImage.LoadFromStream(RS);                
      Image1.Picture.Graphic := JPGImage;
    finally
      RS.Free;
      //MS.Free;
    end;
  finally
    JPGImage.Free;
  end;
end;

procedure loadPForm;
var abitm: TBitmap; SourceImage: TImage; JPGImage: TJPEGImage;
    RS: TResourceStream;
begin
 Form1:= TForm1.create(self)
 with form1 do begin
  SetBounds(317, 175, 1042, 648);
  Caption := 'New Function tester Prime sets from Digits'
  Color := clBtnFace
  Font.Charset := DEFAULT_CHARSET
  Font.Color := clWindowText
  Font.Height := -14
  Font.Name := 'MS Sans Serif'
  Font.Style := []
  OldCreateOrder := False
  OnActivate := @TForm1FormActivate
  PixelsPerInch := 120
  abitm:= TBitmap.create;
  abitm.loadfromresourcename(0, 'MOON_FULL');
  show;
  canvas.brush.bitmap:= abitm;
  Canvas.FillRect(Rect(400,400,200,200));
  //Show;
  TForm1FormActivate(self);
  //TextHeight := 16
  Panel1:= TPanel.create(form1)
  with panel1 do begin
   parent:= form1;
    SetBounds(0,0, 1024, 580);
    Align := alClient
    TabOrder := 0
    aMemo1:= TMemo.create(form1)
    with amemo1 do begin
     parent:= panel1
      Left := 23;  Top := 16
      Width := 336; Height := 281
      Color := 14483455
      Lines.add (
        'Primes from digits:'
        +''
        +'Here'#39's another program answering fairly meaningless '
        +'questions but providing a good programming exercise.  '
        +'' +CRLF+CRLF+
        +'Question  1:'
        +' How many ways can digits 1 through 9 be rearranged to '
        +'form 9 digit prime numbers?'
        +'' +CRLF+CRLF+
        +'Question  2:'
        +'Using the dgits 1 thorugh 9, how many ways can '
        +'they be arranged into sets of numbers containing only  '
        +'primes? (Each concatenation of N digits is treated as an'
        +'N-digit  number.)  For example, one of the  qualifying '
        + 'sets for partition [1,2,2,2,2] is the prime set : {5, 23, 47, 61' +
         + ', '
        +'89}');
      TabOrder := 0
    end;
     image:= TImage.create(form1)
     with image do begin
      parent:= panel1;
      color:= clyellow;
      setbounds(220,300,400,400)
      picture.bitmap.loadfromresourcename(0, 'MOON_FULL');
     end; 
     image1:= TImage.create(form1)
     sourceimage:= TImage.create(form1)
     JpegImage := TJpegImage.Create;
     //JpegImage.loadfromresourcename
    // SourceImage.Picture.LoadFromFile(FullFileName);
      SourceImage.Picture.bitmap.LoadFromresourcename(0, 'MAXBOXLOGO');
      //sourceimage.free;
      //LoadJPEGResource(image1, 'MAXBOX4LOGO'); 
      //storeRCDATAResourcetofile('MAXBOX4LOGO', exepath+'maxbox4logo.jpg');
      writeln(ResourceNameToString('MAXBOX4LOGO'));
      writeln(ResourceTypeToString('JPEG'))
     aMemo3:= TMemo.create(form1)
      //aMemo3:= TMemo.create(form1)
     with amemo3 do begin
      parent:= panel1;
      Left := 832; Top := 16
      Width := 185; Height := 496
      ScrollBars := ssVertical
      TabOrder := 4
     end;
     with image1 do begin
      parent:= panel1;
      color:= clyellow;
      setbounds(5,500,450,400)
      picture.bitmap.loadfromresourcename(0, 'MAXBOXLOGO');
      //JpegImage.Assign(SourceImage.Picture.Graphic);
      //SourceImage.Picture.Bitmap.Assign(JpegImage);
      amemo3.lines.add('Res image 1 loaded')
     end; 
     //TForm5Image1Click(self, image1)
    //LoadJPEGResource(image1, 'MAXBOX4LOGO'); 
     sourceimage.free;
     image2:= TImage.create(form1)
     with image2 do begin
      parent:= panel1;
      color:= clyellow;
      setbounds(450,500,400,400)
      picture.bitmap.loadfromresourcename(0, 'MAXBOXLOGO');
      //JpegImage.Assign(SourceImage.Picture.Graphic);
      //SourceImage.Picture.Bitmap.Assign(JpegImage);
       amemo3.lines.add('Res image 2 logo loaded')
     end; 
     TForm5Image1Click(self, image1)
     image3:= TImage.create(form1)
     with image3 do begin
      parent:= panel1;
      color:= clyellow;
      setbounds(310,100,300,300)
      //SourceImage.Picture.Bitmap.Assign(JpegImage);
      JPGImage:= TJPEGImage.Create;
      RS:= TResourceStream.Create(hInstance,'TRUTH','JPEG');        
      try
        JPGImage.LoadFromStream(RS);                
        Picture.Graphic:= JPGImage;
      finally
        RS.Free;
        JPGImage.Free;
       end;
       amemo3.lines.add('Res image 3 truth loaded') 
     end;
     amemo3.lines.add('Res image 4 moon loaded')   
     //TForm5Image1Click(self, image1)
     image4:= TImage.create(form1)
     with image4 do begin
      parent:= panel1; //panel1;
      color:= clyellow;
      setbounds(5,300,300,300)
      picture.bitmap.loadfromresourcename(0, 'EARTH');
      //JpegImage.Assign(SourceImage.Picture.Graphic);
      //SourceImage.Picture.Bitmap.Assign(JpegImage);
       amemo3.lines.add('Res image 4 EARTH logo loaded')
     end; 
     part1Btn:= TButton.create(form1)
     with part1btn do begin
      parent:= panel1;
      Left := 384; Top := 40
      Width := 161; Height := 31
      Caption := 'Answer Question  1'
      TabOrder := 1
      OnClick := @TForm1Part1BtnClick
    end;
    part2Btn:= TButton.create(form1)
     with part2btn do begin
      parent:= panel1;
      Left := 384
      Top := 80
      Width := 161
      Height := 31
      Caption := 'Answer Question 2'
      TabOrder := 2
      OnClick := @Tform1Part2BtnClick;
    end;
    aMemo2:= TMemo.create(form1)
    with amemo2 do begin
     parent:= panel1;
      Left := 585
      Top := 17
      Width := 232
      Height := 496
      Lines.add(
        'The "Answer Question  2" button '
        +'will show here a summary of prime '
        +'sets for each possible partitioning '
        +'for each arrangement  of  the digits '
        +'1 through 9.'
        +''+CRLF+
        +'Click on any summary line to see '
        +'the prime sets for that partitioning.  '
        +''+CRLF+CRLF+
        +'For each partition clicked, a '
        +'maximum of 100 sets will be '
        +'displayed in the area at right.')
      ScrollBars := ssVertical
      TabOrder := 3
      OnClick := @TForm1Memo2Click
    end;
  end;
  StaticText1:= TStaticText.create(form1)
  with statictext1 do begin
   parent:= form1;
    Left := 0
    Top := 580
    Width := 1024
    Height := 20
    Cursor := crHandPoint
    Align := alBottom
    Alignment := taCenter
    Caption := 'maXbox4 new tester functions V 4.7.6.10'
    Font.Charset := DEFAULT_CHARSET
    Font.Color := clBlue
    Font.Height := -17
    Font.Name := 'Arial'
    Font.Style := [fsBold, fsUnderline]
    ParentFont := False
    TabOrder := 1
    OnClick := @Tform1StaticText1Click;
   end;
 end;
  form1.canvas.brush.bitmap:= abitm;
  form1.Canvas.FillRect(Rect(900,500,500,500));
end; 

procedure TForm1StaticText1Click(Sender: TObject);
//var mbv: TCustomBitmap;
begin
   //ShellExecute(Handle, 'open', 'http://www.delphiforfun.org/',
  //nil, nil, SW_SHOWNORMAL) ;
end;

procedure playWAVRes;
var
  hFind, hRes: THandle;
  Song       : string;
begin
  hFind := FindResource(HInstance, 'GHOST', 'WAV');
  if (hFind <> 0) then begin
    hRes := LoadResource(HInstance, hFind);
    if (hRes <> 0) then begin
      Song := LockResource(hRes);
      if Assigned(Song) then begin
        SndPlaySound(Song, snd_ASync or snd_Memory);
      end;
      UnlockResource(hRes);
    end;
    FreeResource(hFind);
  end;
end;  

procedure TForm2Button1Click(Sender: TObject);
var
  Res: TResourceStream;
begin
  Res := TResourceStream.Create(HInstance, 'APPLAUSE', 'RC_DATA');
  try
    Res.Position := 0;
    //SndPlaySound(Res.Memory, SND_MEMORY or SND_ASYNC or SND_LOOP);
  finally
    Res.Free;
  end;
end;

{Docu of main Neural Unit for maXbox: http://www.softwareschule.ch/examples/uPSI_NeuralNetworkCAI.txt
http://www.softwareschule.ch/examples/uPSI_neuralnetworkcai.txt
http://www.softwareschule.ch/examples/uPSI_neuralvolume.txt
http://www.softwareschule.ch/examples/uPSI_neuraldatasets.txt
http://www.softwareschule.ch/examples/uPSI_neuralfit.txt
Source https://github.com/joaopauloschuler/neural-api }

procedure TTestFitLoadingGetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
    Originals : array of TNeuralFloat; orig2: TByteArray; //array of byte;
    NN: THistoricalNets;
  begin
    LocalX := Random(100);
    LocalY := Random(100);
    Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
    pInput.ReSize(2,1,1);
    //pinput.create1([])
    //pinput.copy41(originals);
    //pinput.copy42(orig2);
    pInput.Data[0,0,0] := LocalX;
    //pInput.FData[1] := LocalY;
    pInput.Data[1,1,1] := LocalY;
    pOutput.ReSize(1,1,1);
    //pOutput.FData[0] := Hypotenuse;
    pOutput.Data[0,0,0] := Hypotenuse;
  end;
  
procedure TTestFitLoadingGetTrainingPair2(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
    Originals : array of TNeuralFloat; orig2: TByteArray; //array of byte;
    NN: THistoricalNets;
    pres: TNeuralFloatArray;
  begin
    LocalX := Random(100);
    LocalY := Random(100);
    Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
    pInput.ReSize(2,1,1);
    //pinput.create1([])
    //pinput.copy41(originals);
    //pinput.copy42(orig2);
    pres:= pinput.fdata;
    writeln(floattostr(pres[0]));
    pInput.Data[0,0,0] := LocalX;
    pres[1]:=  LocalY;
    pInput.FData:= pres;
    //pInput.FData[1] := LocalY;
    pInput.Data[1,1,1] := LocalY;
    pOutput.ReSize(1,1,1);
    //pOutput.FData[0] := Hypotenuse;
    pOutput.Data[0,0,0] := Hypotenuse;
  end;


  procedure TTestFitLoadingGetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    TTestFitLoadingGetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

  procedure TTestFitLoadingGetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    TTestFitLoadingGetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

 //https://github.com/joaopauloschuler/neural-api/blob/master/examples/ImageClassifierSELU/ImageClassifierSELU.lpr
 
 {https://github.com/joaopauloschuler/neural-api/blob/master/examples/ImageClassifierSELU/ImageClassifierSELU.ipynb}
 
 procedure TTestCNNAlgoDoRunClassifier;
 //Application.Title:='CIFAR-10 SELU Classification Example';
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if not CheckCIFARFile() then begin
      //Terminate;
      //exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    //TestDataParallelism( NN);

    NN.AddLayer(TNNetInput.Create4(32, 32, 3) );
    //Function InitSELU( Value : TNeuralFloat) : TNNetLayer');
    NN.AddLayer(TNNetConvolutionLinear.Create(64,5,2,1,1)).InitSELU(0).InitBasicPatterns();
    NN.AddLayer(TNNetMaxPool.Create44(4,0,0) );
    NN.AddLayer(TNNetSELU.Create() );
    NN.AddLayer(TNNetMovingStdNormalization.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1,1,1) ).InitSELU(0);
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1,1,1) ).InitSELU(0);
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1,1,1) ).InitSELU(0);
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1,1,1) ).InitSELU(0);
    NN.AddLayer( TNNetDropout.Create12(0.5,1) );
    NN.AddLayer( TNNetMaxPool.Create44(2,0,0) );
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create28(10,0) );
    NN.AddLayer( TNNetSoftMax.Create() );
    amemo3.lines.add('TNNetConvolutionLinear model add')
    TestDataParallelism( NN);   
    amemo3.lines.add('TestDataParallelism( NN) passed')
    CheckCIFARFile()
    try
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes,
                                          ImgTestVolumes, csEncodeRGB);
    except
      amemo3.lines.add('TNNetConvolutionLinear CIFAR Files missing!')
    end;
    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'ImageClassifierSELU';
    NeuralFit.InitialLearningRate := 0.0004; 
         // SELU seems to work better with smaller learning rates.
    NeuralFit.LearningRateDecay := 0.03;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    //NN.DebugStructure();
    NN.DebugWeights();
    //NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, 
      //           ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    //Terminate;
  end;
  
  
  type TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;
  type TBackOutput = array[0..3] of array[0..2] of TNeuralFloat;

// var  Inputs: TBackInput;
  //    rOutput: TBackOutput;
 
 {
const inputs : TBackInput =
  ( // x1,   x2
    ( 0.1,  0.1), // False, False
    ( 0.1,  0.9), // False, True
    ( 0.9,  0.1), // True,  False
    ( 0.9,  0.9)  // True,  True
  );

const reluoutputs : TBackOutput =
  (// XOR, AND,   OR
    ( 0.1, 0.1, 0.1),
    ( 0.8, 0.1, 0.8),
    ( 0.8, 0.1, 0.8),
    ( 0.1, 0.8, 0.8)
  );  }
  
  procedure definelogicalMatrix(var inputs:TBackInput; var routput:TBackOutput);
  begin
    Inputs[0][0]:= 0.1; Inputs[0][1]:= 0.1;   
    Inputs[1][0]:= 0.1; Inputs[1][1]:= 0.9;   
    Inputs[2][0]:= 0.9; Inputs[2][1]:= 0.1;   
    Inputs[3][0]:= 0.9; Inputs[3][1]:= 0.9;   
    
    routput[0][0]:= 0.1; routput[0][1]:= 0.1;  routput[0][2]:= 0.1;      
    routput[1][0]:= 0.8; routput[1][1]:= 0.1;  routput[1][2]:= 0.8;      
    routput[2][0]:= 0.8; routput[2][1]:= 0.1;  routput[2][2]:= 0.8;      
    routput[3][0]:= 0.1; routput[3][1]:= 0.8;  routput[3][2]:= 0.8;      
   end;
 {   :
  ( 0.1,  0.1), // False, False
    ( 0.1,  0.9), // False, True
    ( 0.9,  0.1), // True,  False
    ( 0.9,  0.9)  // True,  True   }
  
  procedure RunSimpleAlgo();
  var
    NN: TNNet;
    EpochCnt: integer;
    Cnt: integer;
    pOutPut: TNNetVolume;
    vInputs: TBackInput;
    vOutput: TBackOutput;
    inputs: TBackInput;  routput : TBackOutput;
    Rate, Loss, ErrorSum : TNeuralFloat;
  begin
   definelogicalMatrix(inputs, routput);
    NN := TNNet.Create();
    NN.AddLayer( TNNetInput.Create3(2) );
    NN.AddLayer( TNNetFullConnectReLU.Create30(3,0) );
    NN.AddLayer( TNNetFullConnectReLU.Create30(3,0) );       
    NN.SetLearningRate(0.01, 0.9);

    vInputs := inputs;
    vOutput := routput;
  //constructor Create(pSizeX, pSizeY, pDepth: integer; c: T = 0); {$IFNDEF FPC} overload; {$ENDIF}
    pOutPut := TNNetVolume.Create0(3,1,1,1);
    for EpochCnt := 1 to 3000 do begin
      for Cnt := Low(inputs) to High(inputs) do begin
        NN.Compute68(vInputs[Cnt],0);
        NN.GetOutput(pOutPut);
        NN.Backpropagate70(vOutput[Cnt]);
        if EpochCnt mod 300 = 0 then
        WriteLn
        (
          itoa(EpochCnt)+' x '+itoa(Cnt)+
          ' Output:'+
          format(' %5.2f',[poutPut.Raw[0]])+' '+
          format(' %5.2f',[poutPut.Raw[1]])+' '+
          format(' %5.2f',[poutPut.Raw[2]])+' '+
         (* pOutPut.Raw[1]:5:2,' ',
          pOutPut.Raw[2]:5:2, *)
          ' - Training/Desired Output:'+
          format('%5.2f',[vOutput[cnt][0]])+' '+
          format('%5.2f',[vOutput[cnt][1]])+' '+
          format('%5.2f',[vOutput[cnt][2]])+' '
          {vOutput[cnt][0]:5:2,' ',
          vOutput[cnt][1]:5:2,' ' ,
          vOutput[cnt][2]:5:2,' ' }
        );  
      end;
      if EpochCnt mod 300 = 0 then WriteLn('');
    end;
    // TestBatch( NN, pOutPut, 10000,  Rate, Loss, ErrorSum);
     
    //NN.DebugWeights();
    NN.DebugErrors();
    pOutPut.Free;
    NN.Free;
    Write('Press ENTER to exit.');
    amemo3.lines.add('definelogicalMatrix XOR,  AND, & OR solved')
    //ReadLn;
  end;

  
  function TCustomApplicationGetOptionValue(const C: Char; const S: String ): String;

Var
  B : Boolean;
  I : integer;
  capp: TCustomApplication;

begin
  capp:= TCustomApplication.create(self);
  Result:='';
  //I:=TCustomApplicationFindOptionIndex(C,B,0);
  //capp.GetOptionValues
  I:=capp.FindOptionIndex(C,B,0);
   If I=-1 then
    //I:=TCustomApplicationFindOptionIndex(S,B,0);
    I:=capp.FindOptionIndex(S,B,0);
  //If I<>-1 then
    //Result:=capp.GetOptionAtIndex(I,B);
  capp.Free;  
end;

 var asrlist, devicenames, env: TStringlist; abt: boolean;   //GraphicClass: TGraphicClass;
      sessionOptions: OLEVariant;
 
begin //@main

  writeln('CPUspeed: '+cpuspeed+' Hz')
  loadPForm ;
  devicenames:= TStringlist.create;
    //GetWaveOutDevices(DeviceNames);
    GetMIDIOutDevices(devicenames);
   
  if devicenames.count>0 then // only play if wave device present
     //PlaySound('MySound', HINSTANCE, SND_RESOURCE or SND_ASYNC); 
     PlaySound(('SOSUMI_WAV'), hInstance, SND_RESOURCE or SND_ASYNC); 
     writeln(devicenames.text)
   devicenames.free;  //*)
   
   env:= TStringlist.create;
   with TCustomApplication.create(self) do begin
     //run;
     //GetEnvironmentList(env,true)
     //writeln(env.text)
     free
   end;
   env.Free;  
   
   //writeln(botostr(PlayWaveResource('GHOST.WAV')));
   
   //playWAVRes;
   
   playreswav('ghost','wav');
   playreswav('moon','wav');  //}
   
   TTestCNNAlgoDoRunClassifier;
   //TestConvolutionAPI;
   
   RunSimpleAlgo();
   
   //TestDataParallelism( NN : TNNet)');
   //sessionOptions := CreateOleObject('WinSCP.SessionOptions');
    //writeln('IsObjectActive '+botostr(VariantIsObject(sessionOptions)))
   
  (*asrlist::= TStringlist.create;
  if LoadDFMFile2Strings('C:\maXbox\mX47464\maxbox4\web\mX47520\U_PrimesFromDigits.dfm',
                          asrlist, abt)= 0 then writeln(asrlist.text);
    asrlist.Free;    //*)

End.

//dfgfg kloi con-sole

{-- v3
+++ v4
@@ -1,7 +1,7 @@

-Release Notes maXbox 4.7.5.90 V October 2021 mX47
+Release Notes maXbox 4.7.5.90 VI November 2021 mX47

-Add 16 Units + 4 Tutorials
+Add 18 Units + 5 Tutorials

1423 uPSI_SingleList.pas; TSingleListClass
1424 unit uPSI_AdMeter.pas; Async Professional
@@ -19,8 +19,10 @@
1436 unit uPSI_neuraldatasetsv.pas CAI
1437 uPSI_flcFloats.pas FLC5
1438 unit UBigIntsForFloatV4.pas DFF
+1439 unit uPSI_CustApp.pas Pas2js
+1440 unit uPSI_NeuralNetworkCAI.pas CAI

-Total of Function Calls: 34780
-SHA1: of 4.7.5.90 A1C85C7A69602F40F66C84E91196E25E44E0EC7A
-CRC32: 5BDDA952 30.4 MB (31,923,016 bytes)
+Total of Function Calls: 34807
+SHA1: of 4.7.5.90 96DCDE2028125E00B67E42A801721AC513A5EAFC
+CRC32: BBC3A7E5 30.4 MB (31,974,216 bytes)

Sent from sourceforge.net because you indicated interest in https://sourceforge.net/p/maxbox/news/2021/10/maxbox-47590-released-/}


{procedure SIRegister_TCustomApplication(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TComponent', 'TCustomApplication') do
  with CL.AddClassN(CL.FindClass('TComponent'),'TCustomApplication') do
  begin
    Constructor Create( AOwner : TComponent)');
    Procedure HandleException( Sender : TObject)');
    Procedure Initialize');
    Procedure Run');
    Procedure ShowException( E : Exception)');
    Procedure Terminate');
    Procedure Terminate1( AExitCode : Integer)');
    Function FindOptionIndex( const S : String; var Longopt : Boolean; StartAt : Integer) : Integer');
    Function GetOptionValue( const S : String) : String');
    Function GetOptionValue1( const C : Char; const S : String) : String');
    RegisterMethod('Function GetOptionValues( const C : Char; const S : String) : TStringDynArray');
    Runction HasOption( const S : String) : Boolean');
    RFunction HasOption1( const C : Char; const S : String) : Boolean');
    RegisterMethod('Function CheckOptions( const ShortOptions : String; const Longopts : TStrings; Opts, NonOpts : TStrings; AllErrors : Boolean) : String');
    RegisterMethod('Function CheckOptions1( const ShortOptions : String; const Longopts : array of string; Opts, NonOpts : TStrings; AllErrors : Boolean) : String');
    RegisterMethod('Function CheckOptions2( const ShortOptions : String; const Longopts : TStrings; AllErrors : Boolean) : String');
    RegisterMethod('Function CheckOptions3( const ShortOptions : String; const LongOpts : array of string; AllErrors : Boolean) : String');
    RegisterMethod('Function CheckOptions4( const ShortOptions : String; const LongOpts : String; AllErrors : Boolean) : String');
    RegisterMethod('Function GetNonOptions( const ShortOptions : String; const Longopts : array of string) : TStringDynArray');
    RegisterMethod('Procedure GetNonOptions1( const ShortOptions : String; const Longopts : array of string; NonOptions : TStrings)');
    RegisterMethod('Procedure GetEnvironmentList( List : TStrings; NamesOnly : Boolean)');
    RegisterMethod('Procedure GetEnvironmentList1( List : TStrings)');
    RegisterMethod('Procedure Log( EventType : TEventType; const Msg : String)');
    RegisterMethod('Procedure Log1( EventType : TEventType; const Fmt : String; const Args : array of string)');
    RegisterProperty('ExeName', 'string', iptr);
    RegisterProperty('Terminated', 'Boolean', iptr);
    RegisterProperty('Title', 'string', iptrw);
    RegisterProperty('OnException', 'TExceptionEvent', iptrw);
    RegisterProperty('ConsoleApplication', 'Boolean', iptr);
    RegisterProperty('Location', 'String', iptr);
    RegisterProperty('Params', 'String integer', iptr);
    RegisterProperty('ParamCount', 'Integer', iptr);
    RegisterProperty('EnvironmentVariable', 'String String', iptr);
    RegisterProperty('OptionChar', 'Char', iptrw);
    RegisterProperty('CaseSensitiveOptions', 'Boolean', iptrw);
    RegisterProperty('StopOnException', 'Boolean', iptrw);
    RegisterProperty('ExceptionExitCode', 'Longint', iptrw);
    RegisterProperty('ExceptObject', 'Exception', iptrw);
    RegisterProperty('ExceptObjectJS', 'JSValue', iptrw);
    RegisterProperty('EventLogFilter', 'TEventLogTypes', iptrw);
  end;
end;
}
(*procedure SIRegister_TComboSet(CL: TPSPascalCompiler);
begin
  //with RegClassS(CL,'TObject', 'TComboSet') do
  with CL.AddClassN(CL.FindClass('TObject'),'TComboSet') do begin
    RegisterProperty('Selected', 'TByteArray64', iptrw);
    RegisterProperty('RandomRank', 'int64', iptrw);
    RegisterMethod('Procedure Setup( newR, newN : word; NewCtype : TComboType)');
    RegisterMethod('Function Getnext : boolean');
    RegisterMethod('Function GetNextCombo : boolean');
    RegisterMethod('Function GetNextPermute : boolean');
    RegisterMethod('Function GetNextComboWithRep : Boolean');
    RegisterMethod('Function GetNextPermuteWithRep : Boolean');
    RegisterMethod('Function GetCount : int64');
    RegisterMethod('Function GetR : integer');
    RegisterMethod('Function GetN : integer');
    RegisterMethod('Function GetCtype : TCombotype');
    RegisterMethod('Function GetNumberSubsets( const RPick, Number : word; const ACtype : TComboType) : int64');
    RegisterMethod('Function Binomial( const RPick, Number : integer) : int64');
    RegisterMethod('Function Factorial( const Number : integer) : int64');
    RegisterMethod('Function GetRCombo( const RPick, Number : integer) : int64');
    RegisterMethod('Function GetRepRCombo( const RPick, Number : integer) : int64');
    RegisterMethod('Function GetRPermute( const RPick, Number : integer) : int64');
    RegisterMethod('Function GetRepRPermute( const RPick, Number : integer) : int64');
    RegisterMethod('Procedure SetupR( NewR, NewN : word; NewCtype : TComboType)');
    RegisterMethod('Procedure SetupRFirstLast( NewR, NewN : word; NewCType : TComboType)');
    RegisterMethod('Function IsValidRSequence : boolean');
    RegisterMethod('Function ChangeRDirection : boolean');
    RegisterMethod('Function GetNextPrevR : boolean');
    RegisterMethod('Function NextR : boolean');
    RegisterMethod('Function NextLexRPermute : boolean');
    RegisterMethod('Function NextLexRepRPermute : boolean');
    RegisterMethod('Function NextLexRCombo : boolean');
    RegisterMethod('Function NextLexRepRCombo : boolean');
    RegisterMethod('Function NextCoLexRCombo : boolean');
    RegisterMethod('Function PrevR : boolean');
    RegisterMethod('Function PrevCoLexRCombo : boolean');
    RegisterMethod('Function PrevLexRepRPermute : boolean');
    RegisterMethod('Function PrevLexRPermute : boolean');
    RegisterMethod('Function PrevLexRCombo : boolean');
    RegisterMethod('Function PrevLexRepRCombo : boolean');
    RegisterMethod('Function RankR : int64');
    RegisterMethod('Function RankCoLexRCombo : int64');
    RegisterMethod('Function RankLexRCombo : int64');
    RegisterMethod('Function RankLexRepRCombo : int64');
    RegisterMethod('Function RankLexRPermute : int64');
    RegisterMethod('Function RankLexRepRPermute : int64');
    RegisterMethod('Function UnRankR( const Rank : int64) : boolean');
    RegisterMethod('Function UnRankCoLexRCombo( const Rank : int64) : boolean');
    RegisterMethod('Function UnRankLexRCombo( const Rank : int64) : boolean');
    RegisterMethod('Function UnRankLexRepRCombo( const Rank : int64) : boolean');
    RegisterMethod('Function UnRankLexRPermute( const Rank : int64) : boolean');
    RegisterMethod('Function UnRankLexRepRPermute( const Rank : int64) : boolean');
    RegisterMethod('Function RandomR( const RPick, Number : int64; const NewCtype : TComboType) : Boolean');
    RegisterMethod('Function RandomCoLexRCombo( const RPick, Number : int64) : Boolean');
    RegisterMethod('Function RandomLexRCombo( const RPick, Number : int64) : Boolean');
    RegisterMethod('Function RandomLexRepRCombo( const RPick, Number : int64) : Boolean');
    RegisterMethod('Function RandomLexRPermute( const RPick, Number : int64) : Boolean');
    RegisterMethod('Function RandomLexRepRPermute( const RPick, Number : int64) : Boolean');
  end;
end;*)



  
