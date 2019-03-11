package bxb.f2a

import chisel3._
import chisel3.util._

import bxb.util.{Util}
import bxb.memory.{ReadPort}

class F2aSequencer(b: Int, fWidth: Int, qWidth: Int, aWidth: Int, fAddrWidth: Int, qAddrWidth: Int, aAddrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val control = Output(F2aControl())
    // Q Semaphore Pair Dec interface
    val qRawDec = Output(Bool())
    val qRawZero = Input(Bool())
    // A Semaphore Pair Dec interface
    val aWarDec = Output(Bool())
    val aWarZero = Input(Bool())
    // F Semaphore Pair Dec interface
    val fRawDec = Output(Bool())
    val fRawZero = Input(Bool())

    val writeEnable = Output(Bool())

    val vCount = Input(UInt(fAddrWidth.W)) //TODO: split vCount into H&W
    val fOffset = Input(UInt(fAddrWidth.W))
    val qOffset = Input(UInt(qAddrWidth.W))
    val aOffset = Input(UInt(aAddrWidth.W))

    val fmemRead = Output(UInt(fAddrWidth.W))
    val qmemRead = Output(UInt(qAddrWidth.W))
    val amemWriteAddr = Output(UInt(aAddrWidth.W))
  })
  object State {
    val idle :: doingQuantize :: doingQRead :: Nil = Enum(3)
  }

  val state = RegInit(State.idle)
  val idle = (state === State.idle)
  val doingQuantize = (state === State.doingQuantize)
  val doingQRead = (state === State.doingQRead)

  val syncDecFRaw = RegInit(false.B)
  val syncIncFWar = RegInit(false.B)
  val syncDecQRaw = RegInit(false.B)
  val syncIncQWar = RegInit(false.B)
  val syncDecAWar = RegInit(false.B)
  val syncIncARaw = RegInit(false.B)

  val waitRequired = (io.fRawZero | io.qRawZero | io.aWarZero)

  val vCountLeft = Reg(UInt(fAddrWidth.W)) //TODO: split vCount into H&W
  val vCountMax = RegInit(UInt(fAddrWidth.W),0.U)//TODO: split vCount into H&W
  val countLast = (vCountLeft === 0.U)

  val fAddr = RegInit(UInt(fAddrWidth.W), io.fOffset)
  val qAddr = RegInit(UInt(qAddrWidth.W), io.qOffset)
  val aAddr = RegInit(UInt(aAddrWidth.W), io.aOffset)

  val controlWrite = RegInit(false.B)

  when(~waitRequired) {
    when(idle) {
      state := State.doingQRead
      controlWrite := true.B
      syncDecQRaw := true.B
      vCountMax := io.vCount - 1.U

      fAddr := io.fOffset
      qAddr := io.qOffset
      aAddr := io.aOffset
    }.elsewhen(doingQRead) {
      controlWrite := false.B
      syncDecQRaw := false.B
      syncIncQWar := true.B
      vCountLeft := vCountMax
    }
  }
  when(~waitRequired) {
    when(doingQRead) {
      syncDecFRaw := ~syncDecFRaw
      syncIncQWar := ~syncIncQWar
      state := State.doingQuantize
    }.elsewhen(doingQuantize) {
      fAddr := fAddr + 1.U
      vCountLeft := vCountLeft - 1.U
    }
  }
  when(~waitRequired) {
    when(doingQuantize) {
      syncDecAWar := true.B

      aAddr := aAddr + 1.U
      when(countLast) {
        state := State.idle
        syncIncFWar := ~syncIncFWar
        syncIncARaw := ~syncIncARaw
      }
    }.elsewhen(doingQRead) {
      syncDecAWar := false.B
    }
  }


  io.writeEnable := syncDecAWar

  val ffQInc = RegNext(~syncIncQWar)
  val ffFInc = RegNext(~syncIncFWar)
  val ffAInc = RegNext(~syncIncARaw)
  io.control.syncInc.qWar := ~(syncIncQWar ^ ffQInc)
  io.control.syncInc.fWar := ~(syncIncFWar ^ ffFInc)
  io.control.syncInc.aRaw := ~(syncIncARaw ^ ffAInc)

  val ffADec = RegNext(~syncDecAWar)
  val ffQDec = RegNext(~syncDecQRaw)
  val ffFDec = RegNext(~syncDecFRaw)
  io.aWarDec := syncDecAWar && ffADec
  io.qRawDec := syncDecQRaw && ffQDec
  io.fRawDec := ~(syncDecFRaw ^ ffFDec)

  io.fmemRead := fAddr
  io.qmemRead := qAddr  
  io.amemWriteAddr := aAddr

  io.control.qWe := controlWrite
}

object F2aSequencer {
  def main(args: Array[String]): Unit = {
    println(Util.getVerilog(new F2aSequencer(10,10,10,10,10,10,10)))
  }
}
