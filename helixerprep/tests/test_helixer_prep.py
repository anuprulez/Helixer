import os
import numpy as np
import pytest
import deepdish as dd
import sqlalchemy
from sqlalchemy.orm import sessionmaker

import geenuff
from geenuff.tests.test_geenuff import (setup_data_handler, setup_dummyloci_super_locus,
                                        TransspliceDemoData)
from geenuff.applications.importer import ImportController
from geenuff.base.orm import Genome, Coordinate, Transcribed, Feature
from ..core import helpers
from ..core.orm import Mer
from ..core.mers import MerController
from ..export import numerify
from ..export.numerify import SequenceNumerifier, BasePairAnnotationNumerifier, Stepper
from ..export.exporter import ExportController, CoordinateHandler

TMP_DB = 'testdata/tmp.db'
DUMMYLOCI_DB = 'testdata/dummyloci.sqlite3'
H5_OUT = 'testdata/test.h5'


### helper functions ###
def mk_controllers(source_db, helixer_db=TMP_DB, h5_out=H5_OUT):
    if os.path.exists(helixer_db):
        os.remove(helixer_db)

    mer_controller = MerController(source_db, helixer_db)
    export_controller = ExportController(helixer_db, h5_out)
    return mer_controller, export_controller


### preparation and breakdown ###
@pytest.fixture(scope="session", autouse=True)
def setup_dummy_db(request):
    if not os.getcwd().endswith('HelixerPrep/helixerprep'):
        pytest.exit('Tests need to be run from HelixerPrep/helixerprep directory')
    if os.path.exists(DUMMYLOCI_DB):
        os.remove(DUMMYLOCI_DB)
    sl, controller = setup_dummyloci_super_locus('sqlite:///' + DUMMYLOCI_DB)
    coordinate = controller.latest_genome_handler.data.coordinates[0]
    sl.check_and_fix_structure(coordinate=coordinate, controller=controller)
    controller.insertion_queues.execute_so_far()

    # stuff after yield is going to be executed after all tests are run
    yield None

    # clean up tmp files
    for p in [TMP_DB, H5_OUT]:
        if os.path.exists(p):
            os.remove(p)


def mk_memory_session(db_path='sqlite:///:memory:'):
    engine = sqlalchemy.create_engine(db_path, echo=False)
    geenuff.orm.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def memory_import_fasta(fasta_path):
    controller = ImportController(database_path='sqlite:///:memory:')
    controller.add_sequences(fasta_path)
    coord = controller.latest_genome_handler.data.coordinates[0]
    coord_handler = CoordinateHandler(coord)
    return controller, coord_handler


def setup_dummyloci_for_numerify(simplify=False):
    _, controller = mk_controllers(DUMMYLOCI_DB)
    sess = controller.session
    coordinate = sess.query(geenuff.orm.Coordinate).first()
    coord_handler = CoordinateHandler(coordinate)

    # remove erroneous transcripts and their childran
    if simplify:
        for t in ['x', 'z']:
            transcript = sess.query(Transcribed).\
                             filter(Transcribed.given_name == t).first()
            rm_transcript_and_children(transcript, sess)

    return sess, controller, coord_handler


def setup_simpler_numerifier():
    sess, engine = mk_memory_session()
    genome = geenuff.orm.Genome()
    coord, coord_handler = setup_data_handler(CoordinateHandler,
                                              geenuff.orm.Coordinate,
                                              genome=genome,
                                              sequence='A' * 100,
                                              start=0,
                                              end=100,
                                              seqid='a')
    sl = geenuff.orm.SuperLocus()
    transcript = geenuff.orm.Transcribed(super_locus=sl)
    piece = geenuff.orm.TranscribedPiece(transcribed=transcript, position=0)
    transcribed_feature = geenuff.orm.Feature(start=40,
                                              end=9,
                                              is_plus_strand=False,
                                              type=geenuff.types.TRANSCRIBED,
                                              start_is_biological_start=True,
                                              end_is_biological_end=True,
                                              coordinate=coord)
    piece.features = [transcribed_feature]

    sess.add_all([genome, coord, sl, transcript, piece, transcribed_feature])
    sess.commit()
    return sess, coord_handler


### db import from GeenuFF ###
def test_copy_n_import():
    _, controller = mk_controllers(source_db=DUMMYLOCI_DB)
    super_loci = controller.session.query(geenuff.orm.SuperLocus).all()
    assert len(super_loci) == 1
    sl = super_loci[0]
    assert len(sl.transcribeds) == 3
    assert len(sl.translateds) == 3
    all_features = []
    for transcribed in sl.transcribeds:
        assert len(transcribed.transcribed_pieces) == 1
        piece = transcribed.transcribed_pieces[0]
        for feature in piece.features:
            all_features.append(feature)
        print('{}: {}'.format(transcribed.given_name, [x.type.value for x in piece.features]))
    for translated in sl.translateds:
        print('{}: {}'.format(translated.given_name, [x.type.value for x in translated.features]))

    # if I ever get to collapsing redundant features this will change
    assert len(all_features) == 12


### sequences ###
# testing: counting kmers
def test_count2mers():
    mc = helpers.MerCounter(2)
    sequence = 'AAAA'
    mc.add_sequence(sequence)
    counted = mc.export()
    print(counted)
    assert counted['AA'] == 3

    sequence = 'TTT'
    mc.add_sequence(sequence)
    counted = mc.export()
    assert counted['AA'] == 5

    mc2 = helpers.MerCounter(2)
    seq = 'AAATTT'
    mc2.add_sequence(seq)
    counted = mc2.export()
    non0 = [x for x in counted if counted[x] > 0]
    assert len(non0) == 2
    assert counted['AA'] == 4
    assert counted['AT'] == 1


def test_count_range_of_mers():
    seq = 'atatat'

    genome = Genome()
    coordinate = Coordinate(genome=genome, start=0, end=6, sequence=seq)
    coord_handler = CoordinateHandler(coordinate)

    all_mer_counters = coord_handler.count_mers(1, 6)

    assert len(all_mer_counters) == 6

    # make sure total counts for any mer length, k, equal seq_length - k + 1
    for i in range(len(all_mer_counters)):
        counted = all_mer_counters[i].export()
        assert sum(counted.values()) == 6 - i

    # counting 1-mers, expect 6 x 'A'
    counted = all_mer_counters[0].export()
    assert counted['A'] == 6

    # counting 2-mers, expect (3, 'AT'; 2 'TA')
    counted = all_mer_counters[1].export()
    assert counted['AT'] == 3
    assert counted['TA'] == 2

    # counting 3-mers, expect (4, 'ATA')
    counted = all_mer_counters[2].export()
    assert counted['ATA'] == 4

    # counting 4-mers, expect (2, 'ATAT'; 1, 'TATA')
    counted = all_mer_counters[3].export()
    assert counted['ATAT'] == 2
    assert counted['TATA'] == 1

    # counting 5-mers, expect (2, 'ATATA')
    counted = all_mer_counters[4].export()
    assert counted['ATATA'] == 2

    # counting 6-mer, expect original sequence, uppercase
    counted = all_mer_counters[5].export()
    assert counted['ATATAT'] == 1


#### numerify ####
def test_stepper():
    # evenly divided
    s = Stepper(50, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[0] == (0, 10)
    assert strt_ends[-1] == (40, 50)
    # a bit short
    s = Stepper(49, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 5
    assert strt_ends[-1] == (39, 49)
    # a bit long
    s = Stepper(52, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 6
    assert strt_ends[-1] == (46, 52)
    # very short
    s = Stepper(9, 10)
    strt_ends = list(s.step_to_end())
    assert len(strt_ends) == 1
    assert strt_ends[-1] == (0, 9)


def test_short_sequence_numerify():
    _, coord_handler = memory_import_fasta('testdata/tester.fa')
    numerifier = SequenceNumerifier(coord_handler=coord_handler,
                                    is_plus_strand=True,
                                    max_len=100)
    matrix = numerifier.coord_to_matrices()[0]
    # ATATATAT
    x = [0., 1, 0, 0,
         0., 0, 1, 0]
    expect = np.array(x * 4).reshape((-1, 4))
    assert np.array_equal(expect, matrix)


def test_base_level_annotation_numerify():
    sess, controller, coord_handler = setup_dummyloci_for_numerify(simplify=True)
    numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                              features=coord_handler.data.features,
                                              is_plus_strand=True,
                                              max_len=500)
    nums = numerifier.coord_to_matrices()[0]
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    assert np.array_equal(nums, expect)


def test_numerify_from_gr0():
    sess, controller, coord_handler = setup_dummyloci_for_numerify(simplify=True)
    transcribed = sess.query(Feature).filter(
        Feature.type == geenuff.types.OnSequence(geenuff.types.TRANSCRIBED)
    ).all()
    assert len(transcribed) == 1
    transcribed = transcribed[0]
    coord = coord_handler.data
    # move whole region back by 5 (was 0)
    transcribed.start = coord.start = 4
    coord.sequence = coord.sequence[4:]

    # and now once for ranges
    numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                              features=coord_handler.data.features,
                                              is_plus_strand=True,
                                              max_len=500)
    nums = numerifier.coord_to_matrices()[0]
    # as above (except TSS), then truncate
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[4:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.
    expect = expect[4:, :]
    assert np.array_equal(nums, expect)


def test_sequence_slicing():
    """Tests for coherent output when slicing the 405 bp long dummyloci.
    The correct divisions are already tested in the Stepper test.
    The array format of the individual matrices are tested in
    test_short_sequence_numerify().
    """
    sess, controller, coord_handler = setup_dummyloci_for_numerify()
    seq_numerifier = SequenceNumerifier(coord_handler=coord_handler,
                                        is_plus_strand=True,
                                        max_len=100)
    anno_numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                                   features=coord_handler.data.features,
                                                   is_plus_strand=True,
                                                   max_len=100)
    seq_slices = seq_numerifier.coord_to_matrices()
    anno_slices = anno_numerifier.coord_to_matrices()
    assert len(seq_slices) == len(anno_slices) == 5

    # testing error masks
    expect = np.zeros((405, ), dtype=np.int8)
    # sequence error mask should be empty
    assert np.array_equal(expect, seq_numerifier.error_mask)
    # annotation error mask should reflect faulty exon/CDS ranges
    expect[:111] = 1
    expect[119:] = 1
    assert np.array_equal(expect, anno_numerifier.error_mask)


def test_minus_strand_numerify():
    # setup a very basic -strand locus
    sess, coord_handler = setup_simpler_numerifier()
    numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                              features=coord_handler.data.features,
                                              is_plus_strand=True,
                                              max_len=1000)
    nums = numerifier.coord_to_matrices()[0]
    # first, we should make sure the opposite strand is unmarked when empty
    expect = np.zeros([100, 3], dtype=np.float32)
    assert np.array_equal(nums, expect)

    numerifier = BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                              features=coord_handler.data.features,
                                              is_plus_strand=False,
                                              max_len=1000)
    # and now that we get the expect range on the minus strand,
    # keeping in mind the 40 is inclusive, and the 9, not
    nums = numerifier.coord_to_matrices()[0]

    expect[10:41, 0] = 1.
    expect = np.flip(expect, axis=0)
    assert np.array_equal(nums, expect)

    # sequences on minus strand
    _, coord_handler = memory_import_fasta('testdata/biointerp_loci.fa')
    numerifier = SequenceNumerifier(coord_handler=coord_handler,
                                    is_plus_strand=False,
                                    max_len=20000)
    matrix = numerifier.coord_to_matrices()[0]
    assert matrix.shape == (19900, 4,)

    reverse_complement = helpers.reverse_complement(coord_handler.data.sequence)
    expect = [numerify.AMBIGUITY_DECODE[bp] for bp in reverse_complement]
    expect = np.vstack(expect)
    assert np.array_equal(matrix, expect)


def test_h5_gen():
    sess, controller, coord_handler = setup_dummyloci_for_numerify(simplify=True)
    controller.export(chunk_size=400, shuffle=False, seed='puma')

    input_data = dd.io.load(H5_OUT, group='/input')
    labels = dd.io.load(H5_OUT, group='/labels')
    config = dd.io.load(H5_OUT, group='/config')

    assert type(config) == dict

    # prep anno
    expect = np.zeros([405, 3], dtype=np.float32)
    expect[0:400, 0] = 1.  # set genic/in raw transcript
    expect[10:300, 1] = 1.  # set in transcribed
    expect[100:110, 2] = 1.  # both introns
    expect[120:200, 2] = 1.

    # prep seq
    seqexpect = np.full([405, 4], 0.25)

    assert np.array_equal(input_data[0], seqexpect[:202])
    assert np.array_equal(labels[0], expect[:202])

    assert np.array_equal(input_data[1], seqexpect[202:])
    assert np.array_equal(labels[1], expect[202:])


###############################
###############################
###############################

# rm test and replace with feature by coordinate test for interval
def test_intervaltree():
    controller = construct_slice_controller()
    controller.fill_intervaltrees()
    print(controller.interval_trees.keys())
    for intv in controller.interval_trees["1"]:
        s = '{}:{}, {}'.format(intv.begin, intv.end, intv.data.data)
        print(s)
    # check that one known area has two errors, and one transcription termination site as expected
    intervals = controller.interval_trees['1'][399:406]
    assert len(intervals) == 3
    print(intervals, '...intervals')
    print([x.data.data.type.value for x in intervals])
    errors = [x for x in intervals if x.data.data.type.value == geenuff.types.ERROR]

    assert len(errors) == 2
    transcribeds = [x for x in intervals if x.data.data.type.value == geenuff.types.TRANSCRIBED]

    assert len(transcribeds) == 1
    # check that the major filter functions work
    sls = controller.get_super_loci_frm_slice(seqid='1', start=300, end=405, is_plus_strand=True)
    assert len(sls) == 1
    assert isinstance(list(sls)[0], slicer.SuperLocusHandler)

    features = controller.get_features_from_slice(seqid='1', start=0, end=1, is_plus_strand=True)
    assert len(features) == 3
    starts = [x for x in features if x.data.type.value == geenuff.types.TRANSCRIBED]

    assert len(starts) == 2
    errors = [x for x in features if x.data.type.value == geenuff.types.ERROR]
    assert len(errors) == 1


class TransspliceDemoDataSlice(TransspliceDemoData):
    def __init__(self, sess, engine):
        super().__init__(sess)
        self.core_queue = slicer.CoreQueue(session=sess, engine=engine)
        self.genome = geenuff.orm.Genome()
        self.old_coor = geenuff.orm.Coordinate(genome=self.genome, seqid='a', start=1, end=2000)
        self.slh = slicer.SuperLocusHandler(self.sl)
        self.scribedh = slicer.TranscribedHandler(self.scribed)
        self.scribedfliph = slicer.TranscribedHandler(self.scribedflip)

        self.ti = slicer.TranscriptTrimmer(transcript=self.scribedh,
                                           super_locus=self.slh,
                                           sess=sess,
                                           core_queue=self.core_queue)
        self.tiflip = slicer.TranscriptTrimmer(transcript=self.scribedfliph,
                                               super_locus=self.slh,
                                               sess=sess,
                                               core_queue=self.core_queue)


class SimplestDemoData(object):
    def __init__(self, sess, engine, genome=None):
        self.core_queue = slicer.CoreQueue(session=sess, engine=engine)
        if genome is None:
            self.genome = geenuff.orm.Genome()
        else:
            self.genome = genome
        self.old_coor = geenuff.orm.Coordinate(seqid='a', start=0, end=1000, genome=self.genome)
        # setup 1 transition (encoding e.g. a transcript split across a miss assembled scaffold...):
        # 2) scribedlong - [[CD<-],[->AB]] -> ABCD, -> two transcribed pieces: one forward, one backward
        self.sl, self.slh = setup_data_handler(slicer.SuperLocusHandler, geenuff.orm.SuperLocus)
        self.scribedlong, self.scribedlongh = setup_data_handler(slicer.TranscribedHandler,
                                                                 geenuff.orm.Transcribed,
                                                                 super_locus=self.sl)

        self.tilong = slicer.TranscriptTrimmer(transcript=self.scribedlongh,
                                               super_locus=self.slh,
                                               sess=sess,
                                               core_queue=self.core_queue)

        self.pieceAB = geenuff.orm.TranscribedPiece(position=0)
        self.pieceCD = geenuff.orm.TranscribedPiece(position=1)
        self.scribedlong.transcribed_pieces = [self.pieceAB, self.pieceCD]

        self.fAB = geenuff.orm.Feature(transcribed_pieces=[self.pieceAB],
                                       coordinate=self.old_coor,
                                       start=190,
                                       end=210,
                                       start_is_biological_start=True,
                                       end_is_biological_end=False,
                                       given_name='AB',
                                       is_plus_strand=True,
                                       type=geenuff.types.TRANSCRIBED)

        self.fCD = geenuff.orm.Feature(transcribed_pieces=[self.pieceCD],
                                       coordinate=self.old_coor,
                                       is_plus_strand=False,
                                       start=110,
                                       end=90,
                                       start_is_biological_start=False,
                                       end_is_biological_end=True,
                                       type=geenuff.types.TRANSCRIBED,
                                       given_name='CD')

        self.pieceAB.features.append(self.fAB)
        self.pieceCD.features.append(self.fCD)

        sess.add_all([self.scribedlong, self.pieceAB, self.pieceCD, self.fAB, self.fCD,
                      self.old_coor, self.sl])
        sess.commit()
        self.slh.make_all_handlers()


#redo for numerify
def test_transition_unused_coordinates_detection():
    sess, engine = mk_memory_session()
    d = SimplestDemoData(sess, engine)
    # modify to coordinates with complete contain, should work fine
    genome = d.genome
    new_coords = geenuff.orm.Coordinate(seqid='a', start=0, end=300, genome=genome)
    sess.add(new_coords)
    sess.commit()
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d.core_queue.execute_so_far()
    d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
    d.core_queue.execute_so_far()
    assert d.pieceCD in d.scribedlong.transcribed_pieces  # should now keep original at start
    assert d.pieceAB in d.scribedlong.transcribed_pieces
    # modify to coordinates across tiny slice, include those w/o original features, should work fine
    d = SimplestDemoData(sess, engine)
    new_coords_list = [geenuff.orm.Coordinate(genome=genome, seqid='a', start=185, end=195),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=195, end=205),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=205, end=215)]

    for new_coords in new_coords_list:
        sess.add(new_coords)
        sess.commit()
        print('fw {}, {}'.format(new_coords.id, new_coords))
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
        d.core_queue.execute_so_far()
        print([x.id for x in d.tilong.transcript.data.transcribed_pieces])

    new_coords_list = [geenuff.orm.Coordinate(genome=genome, seqid='a', start=105, end=115),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=95, end=105),
                       geenuff.orm.Coordinate(genome=genome, seqid='a', start=85, end=95)]
    for new_coords in new_coords_list:
        sess.add(new_coords)
        sess.commit()
        print('\nstart mod for coords, - strand', new_coords)
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)
        d.core_queue.execute_so_far()

    assert d.pieceCD in d.scribedlong.transcribed_pieces
    assert d.pieceAB in d.scribedlong.transcribed_pieces

    # try and slice before coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=0, end=10)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice after coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=399, end=410)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    # try and slice between slices where there are no coordinates, should raise error
    d = SimplestDemoData(sess, engine)
    new_coords = geenuff.orm.Coordinate(genome=genome, seqid='a', start=149, end=160)
    sess.add(new_coords)
    sess.commit()
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=True)
    d = SimplestDemoData(sess, engine)
    with pytest.raises(slicer.NoFeaturesInSliceError):
        d.tilong.modify4new_slice(new_coords=new_coords, is_plus_strand=False)

# redo for numerify
def test_slicing_featureless_slice_inside_locus():
    controller = construct_slice_controller()
    controller.fill_intervaltrees()
    genome = controller.get_one_genome()
    slh = controller.super_loci[0]
    transcript = [x for x in slh.data.transcribeds if x.given_name == 'y'][0]
    slices = (('1', 'A' * 40, 0, 40, 'train'),
              ('1', 'A' * 40, 40, 80, 'train'),
              ('1', 'A' * 40, 80, 120, 'train'))
    slices = iter(slices)
    controller._slice_annotations_1way(slices, genome=genome, is_plus_strand=True)

    for piece in transcript.transcribed_pieces:
        print('got piece: {}\n-----------\n'.format(piece))
        for feature in piece.features:
            print('    {}'.format(feature))
    coordinate40 = controller.session.query(geenuff.orm.Coordinate).filter(
        geenuff.orm.Coordinate.start == 40
    ).first()
    features40 = coordinate40.features
    print(features40)

    # x & y -> 2 translated, 2 transcribed each, z -> 2 error
    assert len([x for x in features40 if x.type.value == geenuff.types.CODING]) == 2
    assert len([x for x in features40 if x.type.value == geenuff.types.TRANSCRIBED]) == 2
    assert len(features40) == 5
    assert set([x.type.value for x in features40]) == {
        geenuff.types.CODING,
        geenuff.types.TRANSCRIBED,
        geenuff.types.ERROR
    }


def rm_transcript_and_children(transcript, sess):
    for piece in transcript.transcribed_pieces:
        for feature in piece.features:
            sess.delete(feature)
        sess.delete(piece)
    sess.delete(transcript)
    sess.commit()


# change to test edge case numerification (or do that in another test)
def test_reslice_at_same_spot():
    controller = construct_slice_controller()
    slh = controller.super_loci[0]
    # simplify
    transcripty = [x for x in slh.data.transcribeds if x.given_name == 'y'][0]
    transcriptz = [x for x in slh.data.transcribeds if x.given_name == 'z'][0]
    rm_transcript_and_children(transcripty, controller.session)
    rm_transcript_and_children(transcriptz, controller.session)
    # slice
    controller.fill_intervaltrees()
    print('controller.sess', controller.session)
    slices = (('1', 'A' * 100, 0, 100, 'train'), )
    controller._slice_annotations_1way(iter(slices), controller.get_one_genome(), is_plus_strand=True)
    controller.session.commit()
    old_len = len(controller.session.query(geenuff.orm.TranscribedPiece).all())
    print('used to be {} linkages'.format(old_len))
    controller._slice_annotations_1way(iter(slices), controller.get_one_genome(), is_plus_strand=True)
    controller.session.commit()
    assert old_len == len(controller.session.query(geenuff.orm.TranscribedPiece).all())


# fix slice to matrix
def test_live_slicing():
    sess, coord_handler = setup_simpler_numerifier()

    # base pair annotations on minus strand with slicing
    numerifier = numerify.BasePairAnnotationNumerifier(coord_handler=coord_handler,
                                                       is_plus_strand=False)
    num_list = list(numerifier.slice_to_matrices(max_len=50))

    expect = np.zeros([100, 3], dtype=np.float32)
    expect[10:41, 0] = 1.

    assert np.array_equal(num_list[0], np.flip(expect[50:100], axis=0))
    assert np.array_equal(num_list[1], np.flip(expect[0:50], axis=0))

    # dummyloci sequences slicing
    _, coord_handler = memory_import_fasta('testdata/dummyloci.fa')
    numerifier = numerify.SequenceNumerifier(coord_handler=coord_handler, is_plus_strand=True)
    num_list = list(numerifier.slice_to_matrices(max_len=50))
    print([x.shape for x in num_list])
    # [(50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (50, 4), (27, 4), (28, 4)]
    assert len(num_list) == 9
    for i in range(7):
        assert np.array_equal(num_list[i], np.full([50, 4], 0.25, dtype=np.float32))
    for i in [7, 8]:  # for the last two, just care that they're about the expected size...
        assert np.array_equal(num_list[i][:27], np.full([27, 4], 0.25, dtype=np.float32))
